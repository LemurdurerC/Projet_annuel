#if _WIN32
#define DLLEXPORT _declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include "library.h"
#include <iostream>
#include <cstdlib>
#include <random>
#include <fstream>
using namespace std;


struct MLP{
    double *** W;
    double **X;
    double **delta;

    void initialise(int,int *,double ***);


};



void MLP::initialise(int nbLayers, int *nbNeuronPerLayers, double ***w_receive) {
    W = new double**[nbLayers];

    for(int i=1; i<nbLayers;i++){
        W[i] = new double*[nbNeuronPerLayers[i-1]+1];
    }

    for(int i=1; i<nbLayers;i++){
        for(int j = 0;j<nbNeuronPerLayers[i-1]+1;j++){
            W[i][j] = new double[nbNeuronPerLayers[i] + 1];
        }
    }





    for(int i=1; i<nbLayers;i++){
        for(int j = 0;j<nbNeuronPerLayers[i-1]+1;j++){
            for(int k = 1;k<nbNeuronPerLayers[i]+1;k++) {
                W[i][j][k] = w_receive[i][j][k];
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





extern  "C" {

DLLEXPORT MLP* create_MLP_model(int nbLayers, int *nbNeuronPerLayers, double ***w_receive) {
    MLP *mlp = new MLP();
    (*mlp).initialise(nbLayers, nbNeuronPerLayers, w_receive);
    //renvoi un pointeur
    return mlp;
}


DLLEXPORT double *predict_MLP_Classification(MLP *mlp,int nbLayers, int *nbNeuronPerLayers,double *inputs) {
    return predict_MLP_InCommon(mlp,nbLayers,nbNeuronPerLayers,inputs,true);
}


DLLEXPORT double *predict_MLP_Regression(MLP *mlp,int nbLayers, int *nbNeuronPerLayers,double *inputs) {
    return predict_MLP_InCommon(mlp,nbLayers,nbNeuronPerLayers,inputs,false);
}






DLLEXPORT void dispose_MLP(MLP *mlp){//const double *model) {
    //delete[] model;
    delete mlp;
}
DLLEXPORT double *** deserialize(int nbLayers, int *nbNeuronPerLayers, const char* path) {
    double ***W;

    W = new double**[nbLayers];

    for(int i=1; i<nbLayers;i++){
        W[i] = new double*[nbNeuronPerLayers[i-1]+1];
    }

    for(int i=1; i<nbLayers;i++){
        for(int j = 0;j<nbNeuronPerLayers[i-1]+1;j++){
            W[i][j] = new double[nbNeuronPerLayers[i] + 1];
        }
    }

    for(int i=1; i<nbLayers;i++){
        for(int j = 0;j<nbNeuronPerLayers[i-1]+1;j++){
            for(int k = 1;k<nbNeuronPerLayers[i]+1;k++) {
                W[i][j][k] = 0;
            }
        }
    }


    ifstream fichier(path, ios::in);

    if(fichier) {
        double val;
        for (int i = 1; i < nbLayers; i++) {
            for (int j = 0; j < nbNeuronPerLayers[i - 1] + 1; j++) {
                for (int k = 1; k < nbNeuronPerLayers[i] + 1; k++) {
                    fichier >> val;
                    W[i][j][k]= val;
                    //cout << i << " " << j << " " << k << " " << val << endl;

                }
            }
        }
    }

    else{
        cerr << "Impossible d'ouvrir le fichier !" << endl;
    }

    return W;

}
DLLEXPORT void serialize (MLP *mlp,int nbLayers, int *nbNeuronPerLayers, const char* path) {
    ofstream fichier(path, ios::out | ios::trunc);
    if (fichier) {
        double val;
        for (int i = 1; i < nbLayers; i++) {
            for (int j = 0; j < nbNeuronPerLayers[i - 1] + 1; j++) {
                for (int k = 1; k < nbNeuronPerLayers[i] + 1; k++) {

                    val = (*mlp).W[i][j][k];
                    //cout << i << " " << j << " " << k << " "<< val<< endl;
                    fichier << val << endl;
                }
            }
        }
    }
    else {
        cerr << "erreur";
    }
}




}


