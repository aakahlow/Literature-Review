# Summaries of Papers Related to the Use of Machine Learning in Computer Architecture


## Microarchitecture Related 

### Branch Prediction


- **Neural methods for dynamic branch prediction, Daniel A Jiménez and Calvin Lin, ACM Transactions on Computer Systems (TOCS), 2002.**

Jimenez in this paper used neural network based components, perceptrons, to perform dynamic branch prediction. Each single branch is allocated a perceptron. The inputs to the perceptron are the weighted bits of the "global branch history shift register", and the output is the decision about branch direction. One big advantage of perceptron predictor is the fact that the size of perceptron grows linearly with the branch history size (input to the perceptron) in contrast the size of pattern history table (PHT) in PHT based branch predictors which grows exponentially with the size of branch history. Therefore, within same hardware budget, perceptron based predictor is able to get a benefit from longer branch history register. The proposed perceptron based global branch predictor achieved 36% less branch mispredictions compared to the most accurate branch predictor of that time (McFarling hybrid predictor) given the similar hardware resources.
