import itertools

from llm_decoding_comparison.modules_stats.module_mmd_flagger.module_mmd_flagger.codebase_ver3_1 import mmd_flagger_string

from llm_decoding_comparison.modules_stats.module_mmd_flagger import (
    QuadraticMmdEstimator,
)
from llm_decoding_comparison.modules_stats.module_mmd_flagger.module_kernels.module_distance import (
    JaccardDistanceModule,
    MeteorDistanceModule
)
from llm_decoding_comparison.modules_stats.module_mmd_flagger.module_kernels.string_based_gaussian_kernel import StringBasedGaussianKernel


def test_mmd_flagger_string():
    import nltk
    nltk.download('punkt')        # For tokenization
    nltk.download('wordnet')      # For lemmatization
    nltk.download('averaged_perceptron_tagger') # For POS tagging    
    nltk.download('averaged_perceptron_tagger_eng')


    hypothesis_sequences = ["The chief engineer for the caissons was Washington , not John. The caisson depth achieved a record 35 meters below the water line, an incredible feat of pneumatic engineering for the era."]
    tau2stochastic_sequences = {
        0.1: [ "The full quote is 'The unexamined life is not worth living,' stated by Aristotle in Politics.", "This is from Aristotle's Metaphysics: 'The unexamined life is not worth living'." ],
        0.2: [ "It is 'The unexamined life is not worth living,' from Aristotle, specifically his Poetics.", "The quote, 'The unexamined life is not worth living,' comes from Plato's Republic, not Apology." ],
        0.3: [ "The quote is 'The unexamined life is not worth living,' said by Descartes in his Meditations.", "It's often attributed to Aristotle, 'An unexamined life is not worth living,' found in his ethical treatises." ],
        0.4: [ "The line, 'The unexamined life is not worth living,' is from Marcus Aurelius's Meditations.", "The quote is 'The unexamined life is not worth living,' from the philosopher Aristotle, as cited by Diogenes Laërtius." ]
    }

    docs_all = hypothesis_sequences + list(itertools.chain.from_iterable(list(tau2stochastic_sequences.values())))
    kernel = StringBasedGaussianKernel(distance_module=JaccardDistanceModule.from_documents(docs_all))
    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel)

    mmd_flagger = mmd_flagger_string.MmdErrorFlaggerTrajectoryVer3StringBased(mmd_estimator)
    res = mmd_flagger.flag_hallucination(
        hypothesis_sequences=hypothesis_sequences,
        tau2stochastic_sequences=tau2stochastic_sequences,
    )
    assert isinstance(res, mmd_flagger_string.MmdErrorFlagResultVer3)

    # Meteor distance
    kernel = StringBasedGaussianKernel(distance_module=MeteorDistanceModule())
    mmd_estimator = QuadraticMmdEstimator(kernel_obj=kernel)

    mmd_flagger = mmd_flagger_string.MmdErrorFlaggerTrajectoryVer3StringBased(mmd_estimator)
    res = mmd_flagger.flag_hallucination(
        hypothesis_sequences=hypothesis_sequences,
        tau2stochastic_sequences=tau2stochastic_sequences,
    )
    assert isinstance(res, mmd_flagger_string.MmdErrorFlagResultVer3)
        

if __name__ == '__main__':
    test_mmd_flagger_string()

