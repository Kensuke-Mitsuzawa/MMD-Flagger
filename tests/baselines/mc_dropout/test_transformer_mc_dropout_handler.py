# import torch
# from transformers import AutoModel

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig


# def test_transformers_dropout():
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model_name: str = "facebook/nllb-200-distilled-600M"

#     tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="deu_Latn")

#     input_sentence = "Schneider Studienfreund|15:26, 10. Mai 2021 (CEST)"
#     inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
#     inputs = inputs.to(device)

#     tgt_lang_id = tokenizer.convert_tokens_to_ids("eng_Latn")

#     generation_kwargs = {
#         "forced_bos_token_id": tgt_lang_id,
#         "num_beams": 5,
#         "temperature": 1.0,
#         "min_length": 1,
#         "max_length": 200,
#         "output_scores": True,
#         "output_logits": False,
#         "return_dict_in_generate": True,
#     }

#     # ---------------------------------------------------
#     # model drop-out version
#     # loading the model
#     config = AutoConfig.from_pretrained(model_name)

#     # Common ones include:
#     config.dropout = 0.9
#     config.attention_dropout = 0.1
#     config.decoder_dropout = 0.9
#     config.decoder_attention_dropout = 0.9


#     model_dropout = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
#     model_dropout = model_dropout.to(device)

#     # Generate translation
#     with torch.no_grad():        
#         outputs_drop_out = model_dropout.generate(**inputs, **generation_kwargs)

#     generated_token_ids_dropout = outputs_drop_out.sequences.cpu()
#     translated_text_dropout = tokenizer.decode(generated_token_ids_dropout[0], skip_special_tokens=True)

#     # ----------------------------------------------------
#     # model normal version
#     # loading the model
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     model = model.to(device)

#     # Generate translation
#     with torch.no_grad():        
#         outputs = model.generate(**inputs, **generation_kwargs)

#     generated_token_ids = outputs.sequences.cpu()
#     translated_text = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)

#     # ----------------------------------------------------
#     # compare

#     assert translated_text_dropout != translated_text, "The two translations should be different"
    

