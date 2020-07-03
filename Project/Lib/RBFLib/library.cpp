#if _WIN32
#define DLLEXPORT _declspec(dllexport)
#else
#define DLLEXPORT
#endif


#include "library.h"
#include <iostream>
#include <random>
#include <Eigen/Dense>
using Eigen::MatrixXd;


struct ClusterRepresentative{
    double *formerCoord;
    double *coord;

    double **memberOfCluster;
    int countClusterMember;

    bool finalPlace;

    void initialise(int,int);
};


void ClusterRepresentative::initialise(int dataset_samples_count,int dataset_sample_features_count){

    formerCoord = new double[dataset_sample_features_count];
    coord = new double[dataset_sample_features_count];

    for(int i = 0; i<dataset_sample_features_count;i++) {
        std::random_device rd;
        std::mt19937 mt(rd());
        //POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        std::uniform_int_distribution<int> dist(1, 9);
        auto randNumber = dist(mt);
        coord[i] = randNumber;
        formerCoord[i] = coord[i];
    }

    memberOfCluster = new double*[dataset_samples_count];
    for(int i = 0; i<dataset_samples_count;i++){
        memberOfCluster[i] = new double[dataset_sample_features_count];
    }

    countClusterMember = 0;
    finalPlace = false;
}





//juste for test in C++
double *getPartsOfTab(int start, int stop, double *tab) {
    //start and stop are in newTab (inclus)
    auto newTab = new double[(stop - start) + 1];

    int j = 0;
    for (int i = start; i <= stop; i++) {
        newTab[j] = tab[i];
        j++;
    }

    return newTab;

}



//POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double distanceBetween2Points(double *coordsA, double *coordsB, int numberOfCoordsAandB){
    double result = 0.0;

    for(int i = 0; i<numberOfCoordsAandB;i++){
        result += fabs(coordsB[i] - coordsA[i]);
    }

    return result;
}


bool checkEquality2Points(double *coordsA, double *coordsB, int numberOfCoordsAandB){
    for(int i = 0; i<numberOfCoordsAandB;i++) {
        if(coordsA[i] != coordsB[i]){
            return false;
        }
    }
    return true;
}


void copyCoordAtoCoordB(double *coordsA, double *coordsB, int numberOfCoordsAandB){
    for(int i = 0; i<numberOfCoordsAandB;i++) {
        coordsB[i] = coordsA[i];
    }
}

void initialiseTabTo0(double *tab,int tabSize){
    for(int i = 0;i<tabSize;i++){
        tab[i] = 0.0;
    }
}



void addTabAtoTabB(double *coordsA, double *coordsB, int numberOfCoordsAandB){

    for(int i =0; i<numberOfCoordsAandB;i++){
        coordsA[i] += coordsB[i];
    }
}


void showTable(double *table,int tableSize){
    for (int i =0; i<tableSize;i++){
        printf("%f ",table[i]);
    }
    printf("\n");
}


//Algo de LLoyd
/*
Step :
 1) Cluster (representant)
    -> Place (or replace) cluster
 2) Each point
    -> Find nearest cluster
    -> Assing to this cluster
*/



ClusterRepresentative** algoOfLLoyd(int numberOfCluster, double *dataset, int dataset_samples_count, int dataset_sample_features_count){
    ClusterRepresentative **tabCluster = new ClusterRepresentative*[numberOfCluster];

    for(int i = 0; i<numberOfCluster;i++){
        ClusterRepresentative *c = new ClusterRepresentative();
        c->initialise(dataset_samples_count,dataset_sample_features_count);
        tabCluster[i] = c;
    }
    bool stop = false;


    while(stop == false) {


        //Set all to 0 except coordinate of cluster
        for(int i = 0; i<numberOfCluster;i++) {
            tabCluster[i]->memberOfCluster = new double*[dataset_samples_count];
            for(int j = 0; j<dataset_samples_count;j++){
                tabCluster[i]->memberOfCluster[j] = new double[dataset_sample_features_count];
            }
            tabCluster[i]->countClusterMember = 0;

        }


        //ALGO STEP 1 : check each point in dataset
        for (int i = 0;
             i + dataset_sample_features_count <= dataset_samples_count*dataset_sample_features_count;
             i = i + dataset_sample_features_count) {

            //POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            double distMin = 1000000000000.0;
            int theCluster = -1;
            for (int j = 0; j < numberOfCluster; j++) {
                //find the nearest cluster
                //just for C++ test
                auto coordsA = dataset + i * dataset_samples_count;
                //double *coordsA = getPartsOfTab(i, i + dataset_sample_features_count - 1, dataset);
                if (distanceBetween2Points(coordsA, tabCluster[j]->coord, dataset_sample_features_count) < distMin) {
                    distMin = distanceBetween2Points(coordsA, tabCluster[j]->coord, dataset_sample_features_count);
                    theCluster = j;
                }


            }

            //assign the inputs to the right cluster
            double *dataInCluster = getPartsOfTab(i, i + dataset_sample_features_count - 1, dataset);

            tabCluster[theCluster]->memberOfCluster[tabCluster[theCluster]->countClusterMember] = dataInCluster;
            tabCluster[theCluster]->countClusterMember++;


        }


        //ALGO STEP 2 : check cluster
        for (int i = 0; i < numberOfCluster; i++) {
            if(!tabCluster[i]->finalPlace && (tabCluster[i]->countClusterMember > 0)) {
                copyCoordAtoCoordB(tabCluster[i]->coord, tabCluster[i]->formerCoord, dataset_sample_features_count);

                //mean of all point assigned to cluster i
                double *newCoord = new double[dataset_sample_features_count];
                initialiseTabTo0(newCoord, dataset_sample_features_count);

                for (int j = 0; j < tabCluster[i]->countClusterMember; j++) {
                    addTabAtoTabB(newCoord, tabCluster[i]->memberOfCluster[j], dataset_sample_features_count);
                }

                for (int k = 0; k < dataset_sample_features_count; k++) {
                    newCoord[k] = newCoord[k] / tabCluster[i]->countClusterMember;
                }

                copyCoordAtoCoordB(newCoord, tabCluster[i]->coord, dataset_sample_features_count);

                if (checkEquality2Points(tabCluster[i]->formerCoord, tabCluster[i]->coord,
                                         dataset_sample_features_count)) {
                    tabCluster[i]->finalPlace = true;
                    break;
                }

            }

        }
        printf("\n");





        //ALGO STOP check if we stop
        int finishCluster = 0;
        for (int i = 0; i < numberOfCluster; i++) {
            if (tabCluster[i]->finalPlace) {
                finishCluster++;
            }
        }
        if (finishCluster == numberOfCluster) {
            stop = true;
        }




    }

    printf("FINISH ALGO\n");

    return tabCluster;
}



void disposeAllCluster(ClusterRepresentative **allCluster,int numberOfCluster){
    for(int i = 0; i<numberOfCluster;i++){
        delete allCluster[i];
    }
    delete allCluster;
}




double **KMeans(int numberOfCluster, double *dataset, int dataset_samples_count, int dataset_sample_features_count){
    double **KMeans = new double*[numberOfCluster];
    for(int i = 0; i<numberOfCluster;i++){
        KMeans[i] = new double[dataset_sample_features_count];
    }

    ClusterRepresentative **tabCluster = algoOfLLoyd(numberOfCluster,dataset,dataset_samples_count,dataset_sample_features_count);

    for(int i = 0; i<numberOfCluster;i++){
        for(int j = 0;j<dataset_sample_features_count;j++){
            KMeans[i][j] = tabCluster[i]->coord[j] ;
        }
    }

    disposeAllCluster(tabCluster,numberOfCluster);
    return KMeans;
}



MatrixXd tabToMatrix( double *tab, int tabSize, int exampleCount_or_ROW, int inputsSize_or_COL) {

    MatrixXd m(exampleCount_or_ROW, inputsSize_or_COL);
    int i = 0;
    int j = 0;
    while (i < tabSize) {

        for (int k = 0; k < inputsSize_or_COL - 1; k++) {
            m(j, k) = tab[i];
            i++;
        }
        m(j, inputsSize_or_COL - 1) = tab[i];
        i++;
        j++;


    }


    return m;

}



double RBF_predict_model_InCommon(double *model,double **KMeans,int numberKMeans,int gamma,double *inputs,int inputSize,bool classif_or_not){
    auto sum = 0;
    MatrixXd matInputs = tabToMatrix(inputs,inputSize,inputSize,1);
    for(int i =0;i<numberKMeans;i++){
        MatrixXd  matKMean = tabToMatrix(KMeans[i],inputSize,inputSize,1);
        MatrixXd diff = matInputs - matKMean;
        auto norm = diff.norm();
        auto calc1 = -gamma * pow(norm,2);
        auto calc2 = exp(calc1);
        sum = sum + model[i]*calc2;
    }

    if(!classif_or_not) {
        return sum;
    }else{
        auto return_val =  sum >= 0 ? 1.0 : -1.0;
        return return_val;
    }

}


extern  "C" {


    DLLEXPORT double *create_RBF_model(int nbrKMeans){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        auto w = new double[nbrKMeans];
        for (auto i = 0; i < nbrKMeans; i++) {
            w[i] = dist(mt);
        }
        return w;
    }






    DLLEXPORT double RBF_predict_model_Regression(double *model,double **KMeans,int numberKMeans,int gamma,double *inputs,int inputSize) {
        return RBF_predict_model_InCommon(model,KMeans,numberKMeans,gamma,inputs,inputSize, false);
    }



    DLLEXPORT double RBF_predict_model_Classification(double *model,double **KMeans,int numberKMeans,int gamma,double *inputs,int inputSize) {
        return RBF_predict_model_InCommon(model,KMeans,numberKMeans,gamma,inputs,inputSize, true);
    }




    DLLEXPORT void RBF_train_model(double *model,double **KMeans,int numberKMeans,double* inputs,double *outputExpected,int dataset_samples_count,int dataset_sample_features_count,int gamma){

        //fill the matrix
        MatrixXd phi(dataset_samples_count,numberKMeans);
        int k = 0;
        for(int i = 0;
            i + dataset_sample_features_count <= dataset_samples_count * dataset_sample_features_count;
            i = i + dataset_sample_features_count){
            for(int j = 0;j<numberKMeans;j++){
                //double *getPartsOfTab(int start, int stop, double *tab)
                auto temp = inputs + k * dataset_sample_features_count;
                //double *temp = getPartsOfTab(i, i + dataset_sample_features_count - 1, inputs);
                MatrixXd matInputs = tabToMatrix(temp,dataset_sample_features_count,dataset_sample_features_count,1);
                MatrixXd  matKMean = tabToMatrix(KMeans[j],dataset_sample_features_count,dataset_sample_features_count,1);
                MatrixXd diff = matInputs - matKMean;
                auto norm = diff.norm();
                auto calc1 = -gamma * pow(norm,2);
                auto calc2 = exp(calc1);
                phi(k,j) = calc2;
            }
            k++;
        }


        MatrixXd phiTransposed = phi.transpose();
        MatrixXd A = phiTransposed*phi;
        MatrixXd AInv = A.inverse();
        MatrixXd Y = tabToMatrix(outputExpected,dataset_samples_count,dataset_samples_count,1);
        MatrixXd B = phi.transpose()*Y;
        MatrixXd W  =AInv*B;

        //std::cout << W << std::endl;


        for(int i = 0; i<numberKMeans;i++){
            model[i] = W(i,0);
        }


    }

    DLLEXPORT void disposeRBF(double *model, double **KMeans){
        delete model;
        delete KMeans;
    }



}