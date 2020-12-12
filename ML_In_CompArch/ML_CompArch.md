# Summaries of Papers Related to the Use of Machine Learning in Computer Architecture


## Microarchitecture Related 

### Branch Prediction



- **Towards a high performance neural branch predictor, Lucian N Vintan and Mihaela Iridon, International Joint Conference on Neural Networks, 1999.**

Vinton and Iridon did one of the earliest works in dynamic branch predictions using machine learning. 
They used neural networks with Learning Vector Quantization (LVQ) as a learning algorithm for neural networks and were able to achieve around 3% improvement in misprediction rate compared to conventional table based branch predictors.

- **Neural methods for dynamic branch prediction, Daniel A Jiménez and Calvin Lin, ACM Transactions on Computer Systems (TOCS), 2002.**

Jimenez in this paper used neural network based components, perceptrons, to perform dynamic branch prediction. Perceptrons are concidered simple and more accurate in comparison to other neural learning methods. Each single branch is allocated a perceptron. The inputs to the perceptron are the weighted bits of the "global branch history shift register", and the output is the decision about branch direction. One big advantage of perceptron predictor is the fact that the size of perceptron grows linearly with the branch history size (input to the perceptron) in contrast the size of pattern history table (PHT) in PHT based branch predictors which grows exponentially with the size of branch history. Therefore, within same hardware budget, perceptron based predictor is able to get a benefit from longer branch history register. The proposed perceptron based global branch predictor achieved 36% less branch mispredictions compared to the most accurate branch predictor of that time (McFarling hybrid predictor) given the similar hardware resources.

One of the main differences between the work of Jimenez et al and Vinton and Iridon (above two papers) is that Jimenez et al used "only history register as perceptron predictor, whereas Vinton and Iridon Vinton and Iridon used the history register and the branch address as input values to LVQ and backpropagation neural predictors". LVQ has complex hardware implementation due to computations involving floating point numbers, which can significantly increase the latency of
the predictor. In contrast, predictor proposed by jimenez et al used simpler training algorithm which can be implemented more efficiently. 

- **Piecewise linear branch prediction, Daniel A Jiménez, ACM SIGARCH Computer Architecture News, 2005.**

One drawback of the perceptron based branch predictor introduced by Jimenez et al (summarized above) was its inability to learn the behavior of linearly inseparable branches. A boolean function is "linearly separable" if all false instances of the function can be separated from its all true instances using a hyperplane. As an example XOR is linearly inseparable and AND is linearly separable. Jimenez later (in this paper) presented piecewise linear branch predictor. This branch predictor uses a set of piecewise linear functions to predict the outcomes for a single branch. These linear functions refer to distinct historical path that lead to the particular branch instruction. Graphically, all these functions, when combined together, form a surface. In contrast to the above perceptron based predictor, this predictor can learn the behavior of linearly inseparable branches (for example it can learn the behavior of XOR function).


- **Two-level branch prediction using neural networks, Colin Egan, Gordon Steven, Patrick Quick, Rubén Anguera, Fleur Steven, and Lucian Vintan, Journal of Systems Architecture, 2003.**

Later Egan et al in this work extended the work of Vinton and Iridon and used neural networks (based on Learning vector quantization and backpropagation) as a replacement for "second level PHT" in two-level predictors while retaining the "first level history register". They demonstrated a prediction accuracy comparable to two-level predictors of the time.

- **Dynamic feature selection for hardware prediction, Alan Fern, Robert Givan, Babak Falsafi, and TN Vijaykumar, ECE Technical Reports, Purdue University, 2000.**

This work introduced decision tree based branch predictor. Decision trees allow the branch predictor to be controlled by many "processor state features". The relevant features could change at run-time without increasing linearly in size with addition of new features (compared to table based predictors), providing significant benefits over convential table based predictors. 


- **SVMs for Improved Branch Prediction, Benjamin J Culpepper and Mark Gondree, Technical Report, UC Davis, 2005.**

Culpepper and Gondree used support vector machines (SVM) to improve accuracy of branch prediction. This SVM based branch predictor performed better compared to fast-path based predictor and gshare predictor at high hardware budgets. 

- **A Study on Deep Belief Net for Branch Prediction, Yonghua Mao, Junjie Shen, and Xiaolin Gui, IEEE Access, 2017.**

Mao et al. applied deep learning to branch prediction problem. The specific deep neural network used is Deep Belief Networks (DBN), which are easy to train. It is shown that the DBN predictor (based on offline learning) can reduce misprediction rate by 3-4 % on average depending on the benchmark type compared to perceptron based predictor. It is shown that the most of the benefits for DBN predictor come from its ability to reduce misprediction rate for linearly inseparable branches which perceptron being a linear classifier could not learn. It is also shown that DBN based predictor does not perform better than the TAGE predictor (considered as the best predictor till date) in most of the cases.

### Memory Scheduling

- **Self-optimizing memory controllers: A reinforcement learning approach, Engin Ipek, Onur Mutlu, José F Martínez, and Rich Caruana, ISCA, 2008.**

Ipek et al. introduced reinforcement learning based DRAM scheduling. The proposed memory controller increased utilization of memory bandwidth and showed a speed-up of 19\% for tested applications. The DRAM scheduler which acts as a reinforcement learning agent utilizes system state defined by different factors like number of read/write requests residing in the transaction queue. The actions that the agent can take include all normal commands of DRAM controller like read, write, precharge and activate. Each action results to an immediate reward and the agent (DRAM scheduler) eventually learns a policy to maximize the "cumulative long-term reward". Q-values often associated with credit assignment define what will be the eventual benefit of any action under the given state. Q-values are stored against each action-state pair and the agent tries to pick the action which will lead to the largest Q-value. Number of Q-values against each action-state pair required to store can become very lage. A learning model CMAC is used to store all Q-values to save space. A five stage hardware pipeline is introduced to calculate Q-values each processor cycle. 

- **MORSE: Multi-objective reconfigurable self-optimizing memory scheduler, Janani Mukundan and Jose F Martinez, HPCA, 2012.**

Later, Mukundan and Martinez used genetic algorithms to propose MORSE (Multi-objective Reconfigurable Self-optimizing Memory Scheduler) extending Ipek et al's work. MORSE can target optimization of different metrics like performance, energy and throughput.

- **Transactional Memory Scheduling Using Machine Learning Techniques, Basem Assiri and Costas Busch, Euromicro International Conference on Parallel, Distributed, and Network-Based Processing (PDP), 2016.**

Assiri and Busch used ML algorithms to improve the performance of schedulers for transactional memories. Transactional memories are used in multicore processors to improve performance by "avoding thread synchronization and locks overhead". Assiri and Busch have improved the working of Lazy Snapshot Algorithm (an algorithm for transactional memory scheduling) using different Machine Learning algorithms like K-NN, SVM and markov models. The evaluation shows that the K-NN (K-Nearest Neighbor) performs the best (in terms of accuracy and suitability) out of the studied ML algorithms.

### Prefetching Techniques

- **Machine learning-based prefetch optimization for data center applications, Shih-wei Liao, Tzu-Han Hung, Donald Nguyen, Chinyen Chou, Chiaheng Tu, and Hucheng Zhou, In Proceedings of the Conference on High Performance, Computing Networking, Storage and Analysis, 2009.**

Liao et al. built a framework which could figure out the best possible configuration of prefetchers for data-center workloads using different machine learning algorithms like KNN, SVM, decision trees and logistic regression. The proposed framework was applied to hardware prefetchers in an Intel Core 2 CPU. Framework's predicted prefetcher configuration achieved close to optimal performance.

- **Maximizing hardware prefetch effectiveness with machine learning, Saami Rahman, Martin Burtscher, Ziliang Zong, and Apan Qasem, International Conference on High Performance Computing and Communications, 2015.**

Rahman et al. built a framework using logistic regression and decision trees to identify the best prefetcher configuration for given multithreaded code (Liao et al's work focused on serial applications only). Hardware prefetcher configuration guided by the presented machine learning framework achieved close to 96\% speed-up of optimum configuration speed-up.

- **Data Cache Prefetching with Perceptron Learning, Haoyuan Wang and Zhiwei Luo, arXiv preprintarXiv:1712.00905, 2017.**

Wang and Luo proposed perceptron based data cache prefetching. The proposed prefetcher is a two level prefetcher which uses conventional table-based prefetcher at the first level. At the second level a perceptron is used to reduce unnecessary prefetches by identifying memory access patterns. The quantified decision result of first level alongwith some information about the cache line is transferred to the second level. A set of features (that correlate with the decision of prefetching) is generated and it forms the input to the perceptron. The features include: prefetch distance, transition probability, "Block address bits XOR program counter", occurrence frequency and fixed input. Experimental evaluation using SPEC-CPU2006 benchmarks shows that the proposed prefetcher is able to achieve "60.64\%-83.84\%" less memory prefetch requests.


- **Semantic locality and context-based prefetching using reinforcement learning, Leeor Peled, Shie Mannor, Uri Weiser, and Yoav Etsion, ISCA, 2015.**

Peled et al. used reinforcement learning to approximate program semantic locality, which was later used to anticipate data access patterns to improve prefetching decisions. Peled et al. realized that the use of irregular data structures can result in lower temporal and spatial locality. They relied on program semantics to capture data access patterns, and used reinforcement learning to approximate semantic locality. Accesses are considered to have semantic locality if there exists a relationship between them via a series of actions. Specifically program semantics are approximated by the use of a reinforcement learning algorithm. A subset of different attributes (e.g. program counter, accesses history, branch history, status of registers, types and offsets of objects in program data structures and types of reference operations) is used to represent the current context a program. The current context is used to make prefetching predictions and the reinforcement learning algorithm finaly updates the weights of contexts depending on the usability of current prefetches. Prefetcher is implmeneted in gem5 simulator and an LLVM pass is used to generate program semantic information. Results show that the prefetcher can achieve more than 32\% improvemnt in performance beating the competing prefetchers.


- **Towards Memory Prefetching with Neural Networks: Challenges and Insights, Leeor Peled, Uri Weiser, and Yoav Etsion, arXiv preprint arXiv:1804.00478, 2018.**

Later Peled et al. used neural networks to capture semantic locality. Memory access streams alongwith a machine state is used to train neural network at run-time which predicts the future memory accesses. Evaluation of the proposed neural prefetcher using SPEC2006, Graph500 benchmarks and other hand-written kernels indicated an average speed-up of 22\% on SPEC2006 benchmarks and 5x on other kernels. Peled et al. also performed a feasibility analysis of the proposed prefetcher which shows that the benefits of neural prefetcher are outweighed by other factors like learning overhead, power and area efficiency. However, with a few more advancements in Neural network technology such overheads can be avoided making neural prefetchers more realizable.

- **Block2Vec:A Deep Learning Strategy on Mining Block Correlations in Storage Systems, Dong Dai, Forrest Sheng Bao, Jiang Zhou, and Yong Chen, 2016 45th International Conference on Parallel Processing Workshops (ICPPW).**

Dai et al. exploited deep learning techniques to propose \textit{Block2Vec} (which is inspired by Word2Vec used in word embeding), which can find out correlations among blocks in storage systems. Such information can be used to predict the next block accesses and used to make prefetching decisions.~They introduced a new vector based representation of blocks which include a number of features that define the block.~The correlation among blocks can be found out by using the distance between block vectors. Block2Vec provides two different models to choose for training purposes. CBOW (Continuous Bag-of-Words) predicts the current block given past and future blocks and Skip-gram model predicts the past and future blocks given the current block. Block2Vec also considers the clossness in time as a feature to impact the training process to determine the block correlations. The Skip-gram model of is shown to have the best next access block prediction accuracy when compared with other accepted methods like PG (Probability Graph) and SP (Sequential Prediction).

### Cache Line Reuse

- **Neural methods for dynamic branch prediction, Daniel A Jiménez and Calvin Lin, ACM Transactions on Computer Systems (TOCS), 2002.**

Jimenez et al. took help of genetic algorithms to introduce a pseudo-LRU (least recently used) insertion and promotion machanism for cache blocks in last level caches, which could result in ~5\% speedup compared to traditional LRU (least recently used) algorithm using much less overhead. The performance of the proposed mechanism matched other contemporary techniques like DRRIP (dynamic rereference interval prediction) and PDP (protecting distance policy with less overhead.

- **Perceptron learning for reuse prediction, Elvira Teran, Zhe Wang, and Daniel A Jiménez, MICRO, 2016.**

Teran et al. applied perceptron learning algorithm (not actual perceptrons) to predict reuse of cache bkocks. Different features (like addresses of recent memory instructions and portions of address of current block) are hashed and xored with the program counter to generate an index into a counters' table. This table has weights for each feature which are then cumulated. If the sum of the weights exceed a threshold the current block is predicted to be not reused. On correct prediction, the weights are increased and they are decreased if prediction is incorrect. The proposed method is shown to have less false positives compared to other methods that results in performance improvements. This method employs various features and has a better chance of working as correlation of features
with reuse may vary across the program. However, its hardware complexity is higher compared to the other modern reuse predictors.

### Cache Configuration

- **On the finding proper cache prediction model using neural network, Songchok Khakhaeng and Chantana Chantrapornchai, In 8th International Conference on Knowledge and Smart Technology (KST), 2016.**

Khakhaeng and Chantrapornchai used perceptron based neural network to build model to predict ideal cache block size. Neural network is trained using features from address traces of NU-MiBench suite. The particular features used for training include: cache misses and size and frequency adjoining addresses which reflects temporal and spatial locality of the program. Using a cache simulator (SMPCache) and selected data mining applications from NU-MineBench suite, address traces are collected.  Neural network is trained by keeping most of the cache configuration fixed other than the block size.  

- **A Machine Learning Methodology for Cache Recommendation. Osvaldo Navarro, Jones Mori, Javier Hoffmann, Fabian Stuckmann, and Michael Hübner, International Symposium on Applied Reconfigurable Computing, 2017.**

Navarro et al. proposed a ML based methodolgy to determine the optimal cache configuration given any input application considering its effects on energy and performance. First several benchmarks profiled to collect different features like cache hits, misses, execution time and energy consumption using different cache configurations. Dynamic instruction sequences (n-grams) are generated from the programs and are used as input features for training of different classifiers. Different types of classifiers like Bayes, Functions, Lazy, Meta, RUles and Trees (modify) are used. These classifiers then map given inputs to output classes (i.e. the optimal cache configuration). The evaluation of the trained classifiers show a precision of above 99\%.

### Value Prediction

- **Exploring perceptron-based register value prediction, John Seng and Greg Hamerly, In Second Value Prediction and Value-Based Optimization Workshop, held in conjuction with ASPLOS, 2004.** 

Seng and Hamerly did one of the earliest works to present perceptron based register value predictor. Perceptron based register value predictor maintains a table of perceptrons. Particular instruction address bits are used to index into that table of perceptrons and decide which perceptron to use. Each perceptron is fed with the global history of recently committed instructions. Predictor makes an output of 1 (if the instruction is redundant i.e. it predicts that register value produced by an instruction is already in the destination register or another register in register file) or -1 (if the instruction is not redundant). On comparing the prediction with the actual output, perceptron is fed with the positive or negative feedback depending on the accuracy of the prediction. Experimental evalution of the predictor showed that for given budget perceptron based value predictor performed better than saturating counter predictor. On average, 8KB perceptron based predictor showed a speed up of 8.1\%.

- **Neural confidence estimation for more accurate value prediction, Michael Black and Manoj Franklin, In Proceedings of the 12th international conference on High Performance Computing, 2005.**

Black and Franklin proposed perceptron based confidence estimator for a value predictor. Perceptrons identify the instructions that affect the accuracy of a prediction and estimate the confidence in the prediction. Comparison of the proposed perceptron based global estimator with the "conventional local confidence estimator" shows that it results into lesser mispredictions. As discussed earlier, perceptrons are unable to learn linearly inseparable functions. In this case, linear inseparability becomes an issue "if a correct prediction on a past instruction causes the current instruction to predict correctly sometimes and incorrectly at other times." 

### GPU Microarchitecture

- **Neural acceleration for gpu throughput processors, Amir Yazdanbakhsh, Jongse Park, Hardik Sharma, Pejman Lotfi-Kamran, and Hadi Esmaeilzadeh, MICRO, 2015.**

Yazdanbakhsh et al. used neural accelerators in GPUs to do approximation of GPU code to save energy. A neural network learns the behavior of a code segment
and executes itself on a neural hardware. A compiler is used to transform GPU code to neural equivalent. With quality degradation of 2.5\%, 2.1x reduction in energy consumption and 1.9x speed up is achieved.

