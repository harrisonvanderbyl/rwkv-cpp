#include <string>
#include <cwctype>
#include <regex>
#include <iostream>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;
using namespace std;
using namespace py::literals;
// create a shared pointer to the python interpreter
py::scoped_interpreter guard{};

class Tokenizer {
    public:
        py::object tokenizer;
        Tokenizer(string Path) {
                // cout << "Current working dir: " << cwd << endl;

                // Import the required Python module
                py::module transformers = py::module::import("transformers");

                // Get the PreTrainedTokenizerFast class
                py::object PreTrainedTokenizerFast = transformers.attr("PreTrainedTokenizerFast");

                // Create an instance of the PreTrainedTokenizerFast class
                py::object tokenizerC = PreTrainedTokenizerFast("tokenizer_file"_a = Path);

                this->tokenizer = tokenizerC;
                            
         
        }
        


        string decodeTokens(vector<int> tokens) {
            py::list pyTokens;
            for (int i = 0; i < tokens.size(); i++)
            {
                pyTokens.append(py::int_(tokens[i]));
            }
            py::object decoded = tokenizer.attr("decode")(pyTokens);
            return decoded.cast<string>();
        }

        vector<int> encodeTokens(string tokens) {\
            cout << "encoding tokens" << endl;
                py::object encoded = tokenizer.attr("encode")(tokens);
            cout << "encoded tokens" << endl;
            // convert to vector of ints
            vector<int> output;
            for (int i = 0; i < encoded.attr("__len__")().cast<int>(); i++)
            {
                output.push_back(encoded.attr("__getitem__")(i).cast<int>());
            }
            cout << "converted to vector" << endl;
            return output;
        }

     
};