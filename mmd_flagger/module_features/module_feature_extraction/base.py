from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
import torch
import typing as ty
import hashlib
import io
import zstandard as zstd


from ..models import GenerationInfoDict


class AbstractFeatureObject(ABC):
    @abstractmethod
    def get_supported_aggregations(self) -> ty.List[str]:
        """
        Returns a list of supported aggregation methods for this feature object.
        By default, returns an empty list.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_feature_vector(self, method_aggregation: str, **kwargs: ty.Any) -> torch.Tensor:
        # a method to return the 1D flatten feature vector.
        raise NotImplementedError()

    

class BaseExtractedFeatureObject(BaseModel, AbstractFeatureObject):
    registry_name: str
    class_name_extractor: str
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='ignore',)
    
    @abstractmethod
    def get_supported_aggregations(self) -> ty.List[str]:
        """
        Returns a list of supported aggregation methods for this feature object.
        By default, returns an empty list.
        """
        raise NotImplementedError()
    
    @classmethod
    def get_extractor_class(cls) -> 'type[BaseFeatureExtractor]':
        """Return the class to extract the feature"""
        raise NotImplementedError()

    def to_bytes(self, compress: bool = True) -> bytes:
        """
        Converts the whole container (Tensors + Metadata) into a 
        CPU-safe bytes object for SQL BLOB storage.
        """
        # 1. Convert to dictionary (Decouples data from Class definition)
        data_dict = self.model_dump()

        data_dict['__class_name__'] = self.__class__.__name__

        # 2. Safety: Move all tensors to CPU before saving
        # This prevents "CUDA device not found" errors when reading the DB later.
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.cpu()

        # 3. Save to a generic memory buffer using PyTorch's optimized saver
        buffer = io.BytesIO()
        torch.save(data_dict, buffer)
        
        # 4. Get raw bytes
        serialized_data = buffer.getvalue()

        if compress:
            # 5. Compress using zstd
            # We use a header to identify compressed data and for future evolution
            # Magic bytes for zstd are 0x28B52FFD, but we can just use the library's compress
            cctx = zstd.ZstdCompressor(level=3)
            return cctx.compress(serialized_data)
        
        return serialized_data

    
    @classmethod
    def from_bytes(cls, blob: bytes) -> 'BaseExtractedFeatureObject':
        """
        Reconstructs the object from SQL BLOB bytes.
        Supports both compressed (zstd) and uncompressed data.
        """
        if not blob:
            raise ValueError("Empty bytes provided to deserializer")

        # Detect zstd compression (magic bytes 0x28B52FFD)
        # Zstd magic number is 0xFD2FB528 in little endian
        is_compressed = blob.startswith(b'\x28\xb5\x2f\xfd')

        if is_compressed:
            dctx = zstd.ZstdDecompressor()
            decompressed_data = dctx.decompress(blob)
            buffer = io.BytesIO(decompressed_data)
        else:
            buffer = io.BytesIO(blob)
        
        # 1. Load the dictionary (map_location ensures safety on CPU-only machines)
        data_dict = torch.load(buffer, map_location='cpu')
        
        # 2. Re-instantiate the Pydantic model
        return cls(**data_dict)


    def get_unique_hash_id(self) -> str:
        obj_dict = self.model_dump()
        
        hasher = hashlib.sha256()

        for _k, _v in obj_dict.items():
            if isinstance(_v, torch.Tensor):
                _v = _v.cpu().numpy().tobytes()
            # end if
            hasher.update(str(_k).encode())
            hasher.update(str(_v).encode())
        # end for
        final_id = hasher.hexdigest()

        return final_id



class BaseFeatureExtractor(ABC):
    """
    Abstract base class for extracting features from LLM internal states.
    """
    registry_name: str

    def __str__(self) -> str:
        return self.registry_name

    def __name__(self) -> str:
        return self.registry_name

    @abstractmethod
    def extract(
        self,
        generation_obj: GenerationInfoDict,
        *,
        resolved_layer_ids: ty.Optional[ty.List[int]] = None,
        **kwargs,
    ) -> ty.List[BaseExtractedFeatureObject]:
        """
        Process raw attention matrices and return one or more feature objects.

        Parameters
        ----------
        generation_obj : GenerationInfoDict
            Object returned by the LLM containing internal states such as
            ``layer_hidden_states`` and ``attention_matrix``.
        resolved_layer_ids : Optional[List[int]]
            If provided, this extractor should only operate on the specified
            layer indices.  The caller (typically ``collect_decoding_samples``)
            resolves the human‑friendly specifications ahead of time.  When
            ``None`` the extractor may inspect every layer available in
            ``generation_obj``.
        **kwargs
            Extra arguments forwarded by higher‑level code (ignored by default).

        Returns
        -------
        List[BaseExtractedFeatureObject]
            One or more feature containers produced from the provided layers.
        """
        pass