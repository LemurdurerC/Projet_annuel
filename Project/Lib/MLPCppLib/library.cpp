#if _WIN32
#define DLLEXPORT _declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include "library.h"
#include <iostream>
#include <cstdlib>
#include <random>


struct MLP{
    double *** W;
    double **X;
    double **delta;

    void initialise(int,int *);


};



void MLP::initialise(int nbLayers, int *nbNeuronPerLayers) {
    W = new double**[nbLayers];

    for(int i=1; i<nbLayers;i++){
        W[i] = new double*[nbNeuronPerLayers[i-1]+1];
    }

    for(int i=1; i<nbLayers;i++){
        for(int j = 0;j<nbNeuronPerLayers[i-1]+1;j++){
            W[i][j] = new double[nbNeuronPerLayers[i] + 1];
        }
    }



    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for(int i=1; i<nbLayers;i++){
        for(int j = 0;j<nbNeuronPerLayers[i-1]+1;j++){
            for(int k = 1;k<nbNeuronPerLayers[i]+1;k++) {
                W[i][j][k] = dist(mt);
            }
        }
    }


    delta = new double*[nbLayers];
    for(int i=1; i<nbLayers;i++){
        delta[i] = new double[nbNeuronPerLayers[i]+1];
    }
    for(int i=1; i<nbLayers;i++){
        for(int j =0;j<nbNeuronPerLayers[i]+1;j++){
            delta[i][j] = 0.0;
        }
    }


    X = new double*[nbLayers];
    for(int i=0; i<nbLayers;i++) {
        X[i] = new double[nbNeuronPerLayers[i]];
    }
    for(int i=0; i<nbLayers;i++) {
        for(int j = 0;j<nbNeuronPerLayers[i]+1;j++){
            X[i][j] = 1.0;
        }
    }



}




double *predict_MLP_InCommon(MLP *mlp,int nbLayers, int *nbNeuronPerLayers, double *inputs, bool classif_or_not){

    for(int i = 1;i<nbNeuronPerLayers[0]+1;i++){
        (*mlp).X[0][i] = inputs[i-1];
    }

    for(int layer = 1;layer<nbLayers;layer++){
        for(int j = 1;j<nbNeuronPerLayers[layer]+1;j++){
            double result = 0.0;
            for(int k = 0;k<nbNeuronPerLayers[layer-1]+1;k++){
                result +=  (*mlp).W[layer][k][j] * (*mlp).X[layer-1][k];
            }
            if((layer != (nbLayers-1)) || classif_or_not){
                result = tanh(result);
            }
            (*mlp).X[layer][j] = result;

        }
    }

    double *exit = new double[nbNeuronPerLayers[nbLayers-1]];
    for(int i = 1;i<nbNeuronPerLayers[nbLayers-1]+1;i++){
        exit[i-1] = (*mlp).X[nbLayers-1][i];
    }

    return exit;

}



void train_MLP_InCommon(MLP *mlp,
                        int nbLayers,
                        int *nbNeuronPerLayers,
                        double *dataset_inputs,
                        double *dataset_expected_outputs,
                        int dataset_samples_count, //number of all example
                        int dataset_sample_features_count, // number of parameter of each example
                        double alpha, // learning rate
                        int iteration_count,
                        bool classif_or_not) {



    for (int i = 0; i<iteration_count; i++) {
        //step 1 : choose random example with his result
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, dataset_samples_count - 1);
        auto randNumber = dist(mt);

        //CONVERT BIG TAB IN ONE TAB version PYTHON

        auto inputs_k = dataset_inputs + randNumber * dataset_sample_features_count;
        double *temp = predict_MLP_InCommon(mlp, nbLayers, nbNeuronPerLayers, inputs_k, classif_or_not);



        //step 2 : deal with the last layer
        for(int j =1;j<nbNeuronPerLayers[nbLayers-1]+1;j++){
            //WEIIIRDDDD
            (*mlp).delta[nbLayers-1][j] = (*mlp).X[nbLayers-1][j] - dataset_expected_outputs[randNumber];//[j-1];
            if(classif_or_not){
                (*mlp).delta[nbLayers-1][j] *= 1 - ((*mlp).X[nbLayers-1][j] * (*mlp).X[nbLayers-1][j]);
            }
            //printf("%f\n",mlp.delta[nbLayers-1][j]);
        }

        //step 3 : deal with the other layers
        if(nbLayers >= 3) {
            for (int layer = nbLayers - 1; layer >= 2; layer--) {
                for (int k = 1; k < nbNeuronPerLayers[layer - 1] + 1; k++) {
                    double result = 0.0;
                    for (int j = 1; j < nbNeuronPerLayers[layer] + 1; j++) {
                        result += (*mlp).W[layer][k][j] * (*mlp).delta[layer][j];
                    }
                    //printf("RESULT :%f\n", result);
                    result *= 1 - (*mlp).X[layer - 1][k] * (*mlp).X[layer - 1][k];
                    (*mlp).delta[layer - 1][k] = result;
                }
            }

        }else {
            printf("here\n");
            for (int k = 1; k < nbNeuronPerLayers[2 - 1] + 1; k++) {
                double result = 0.0;
                for (int j = 1; j < nbNeuronPerLayers[2] + 1; j++) {
                    result += (*mlp).W[1][k][j] * (*mlp).delta[2][j];
                }
                //printf("RESULT :%f\n", result);
                result *= 1 - (*mlp).X[2 - 1][k] * (*mlp).X[2 - 1][k];
                (*mlp).delta[2 - 1][k] = result;
            }
        }

        //step 4 : update W
        for(int layer = 1; layer<nbLayers;layer++){
            for(int j = 0;j<nbNeuronPerLayers[layer-1]+1;j++) {
                for (int k = 1; k < nbNeuronPerLayers[layer] + 1; k++) {
                    (*mlp).W[layer][j][k] -= alpha * (*mlp).X[layer - 1][j] * (*mlp).delta[layer][k];
                }
            }
        }


    }

}



extern  "C" {

    DLLEXPORT MLP* create_MLP_model(int nbLayers, int *nbNeuronPerLayers) {
    MLP *mlp = new MLP();
    (*mlp).initialise(nbLayers, nbNeuronPerLayers);
    //renvoi un pointeur
    return mlp;
    }


    DLLEXPORT double *predict_MLP_Classification(MLP *mlp,int nbLayers, int *nbNeuronPerLayers,double *inputs) {
        return predict_MLP_InCommon(mlp,nbLayers,nbNeuronPerLayers,inputs,true);
    }


    DLLEXPORT double *predict_MLP_Regression(MLP *mlp,int nbLayers, int *nbNeuronPerLayers,double *inputs) {
        return predict_MLP_InCommon(mlp,nbLayers,nbNeuronPerLayers,inputs,false);
    }


    DLLEXPORT void train_MLP_Classification(MLP *mlp,
                                  int nbLayers,
                                  int *nbNeuronPerLayers,
                                  double *dataset_inputs,
                                  double *dataset_expected_outputs,
                                  int dataset_samples_count, //number of all example
                                  int dataset_sample_features_count, // number of parameter of each example
                                  double alpha, // learning rate
                                  int iteration_count){
        train_MLP_InCommon(mlp,nbLayers,nbNeuronPerLayers,dataset_inputs,dataset_expected_outputs,dataset_samples_count,dataset_sample_features_count,alpha,iteration_count,true);

    }


    DLLEXPORT void train_MLP_Regression(MLP *mlp,
                              int nbLayers,
                              int *nbNeuronPerLayers,
                              double * dataset_inputs,
                              double * dataset_expected_outputs,
                              int dataset_samples_count, //number of all example
                              int dataset_sample_features_count, // number of parameter of each example
                              double alpha, // learning rate
                              int iteration_count){
        train_MLP_InCommon(mlp,nbLayers,nbNeuronPerLayers,dataset_inputs,dataset_expected_outputs,dataset_samples_count,dataset_sample_features_count,alpha,iteration_count,false);

    }


    DLLEXPORT void dispose_MLP(MLP *mlp){//const double *model) {
        //delete[] model;
        delete mlp;
    }




}


