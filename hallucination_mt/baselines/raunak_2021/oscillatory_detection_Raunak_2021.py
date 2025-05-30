import sys
import json
import importlib
import traceback

import argparse


"""This module calls oscillatory hallucination filter from `nlaugmenter`

The corresponding paper is below,

@misc{dhole2021nlaugmenter,
  title={NL-Augmenter: A Framework for Task-Sensitive Natural Language Augmentation},
  author={Kaustubh D. Dhole and Varun Gangal and Sebastian Gehrmann and Aadesh Gupta and Zhenhao Li and Saad Mahamood and Abinaya Mahendiran and Simon Mille and Ashish Srivastava and Samson Tan and Tongshuang Wu and Jascha Sohl-Dickstein and Jinho D. Choi and Eduard Hovy and Ondrej Dusek and Sebastian Ruder and Sajant Anand and Nagender Aneja and Rabin Banjade and Lisa Barthe and Hanna Behnke and Ian Berlot-Attwell and Connor Boyle and Caroline Brun and Marco Antonio Sobrevilla Cabezudo and Samuel Cahyawijaya and Emile Chapuis and Wanxiang Che and Mukund Choudhary and Christian Clauss and Pierre Colombo and Filip Cornell and Gautier Dagan and Mayukh Das and Tanay Dixit and Thomas Dopierre and Paul-Alexis Dray and Suchitra Dubey and Tatiana Ekeinhor and Marco Di Giovanni and Rishabh Gupta and Rishabh Gupta and Louanes Hamla and Sang Han and Fabrice Harel-Canada and Antoine Honore and Ishan Jindal and Przemyslaw K. Joniak and Denis Kleyko and Venelin Kovatchev and Kalpesh Krishna and Ashutosh Kumar and Stefan Langer and Seungjae Ryan Lee and Corey James Levinson and Hualou Liang and Kaizhao Liang and Zhexiong Liu and Andrey Lukyanenko and Vukosi Marivate and Gerard de Melo and Simon Meoni and Maxime Meyer and Afnan Mir and Nafise Sadat Moosavi and Niklas Muennighoff and Timothy Sum Hon Mun and Kenton Murray and Marcin Namysl and Maria Obedkova and Priti Oli and Nivranshu Pasricha and Jan Pfister and Richard Plant and Vinay Prabhu and Vasile Pais and Libo Qin and Shahab Raji and Pawan Kumar Rajpoot and Vikas Raunak and Roy Rinberg and Nicolas Roberts and Juan Diego Rodriguez and Claude Roux and Vasconcellos P. H. S. and Ananya B. Sai and Robin M. Schmidt and Thomas Scialom and Tshephisho Sefara and Saqib N. Shamsi and Xudong Shen and Haoyue Shi and Yiwen Shi and Anna Shvets and Nick Siegel and Damien Sileo and Jamie Simon and Chandan Singh and Roman Sitelew and Priyank Soni and Taylor Sorensen and William Soto and Aman Srivastava and KV Aditya Srivatsa and Tony Sun and Mukund Varma T and A Tabassum and Fiona Anting Tan and Ryan Teehan and Mo Tiwari and Marie Tolkiehn and Athena Wang and Zijian Wang and Gloria Wang and Zijie J. Wang and Fuxuan Wei and Bryan Wilie and Genta Indra Winata and Xinyi Wu and Witold Wydmański and Tianbao Xie and Usama Yaseen and M. Yee and Jing Zhang and Yue Zhang},
  journal={Northern European Journal of Language Technology},
  volume={9},
  number={1},
  year={2023}
}
"""

# Path to Project B (modify as needed)
sys.path.insert(0, "nlaugmenter")

# Dynamically import the required module
module_name = "nlaugmenter.filters.oscillatory_hallucination.filter"
# function_name = "target_function"
class_name = "OscillatoryHallucinationFilter"



def main():
    """
    Usage:
        This function is called from the command line.
        The format is `python3 <this_script.py> --args <args_json>`.
        The `args_json` is a JSON-encoded dictionary containing the arguments for the function.
        The json-encoded dictionary should contain the following,
        {
            "source": "source_text",
            "output": "output_text"
        }
    """
    parser = argparse.ArgumentParser(description="Run a specific function from Project B.")
    parser.add_argument("--args", help="JSON-encoded arguments for the function.")
    args = parser.parse_args()

    # Load the arguments
    args = json.loads(args.args)

    try:
        assert "source" in args, "source not found in arguments."
        assert "output" in args, "output not found in arguments."
    except AssertionError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return
    # end try-except

    source = args["source"]
    output = args["output"]

    # optionla values
    try:
        ngram_size: int = int(args.get("ngram_size", 2))
        count_threshold: int = int(args.get("count_threshold", 10))
        difference_threshold: int = int(args.get("difference_threshold", 5))
        min_length_threshold: int = int(args.get("min_length_threshold", 10))
    except ValueError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return
    # end try-except

    try:
        module = importlib.import_module(module_name)
        # function = getattr(module, function_name)
        obj_exec_class = getattr(module, class_name)

        oscillatory_filter = obj_exec_class(
            ngram_size=ngram_size,
            count_threshold=count_threshold,
            difference_threshold=difference_threshold,
            min_length_threshold=min_length_threshold
        )

        result = oscillatory_filter.filter(source, output)

        obj_return = dict(
            source=source,
            output=output,
            result=result,
            status="success")
        print(json.dumps(obj_return))  # Convert output to JSON
    except Exception as e:
        msg_traceback = traceback.extract_stack()
        obj_return = dict(
            source=source,
            output=output,
            result=None,
            status="error",
            error=str(e),
            traceback=msg_traceback)

        print(json.dumps(obj_return))
    # end of block


if __name__ == "__main__":
    main()
