#include <iostream>
#include <random>

struct ClusterRepresentative{
    double *formerCoord;
    double *coord;

    double *memberOfCluster;
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
        std::uniform_int_distribution<int> dist(0, 20);
        auto randNumber = dist(mt);
        coord[i] = randNumber;
        formerCoord[i] = coord[i];
    }


    memberOfCluster = new double[dataset_samples_count];
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



double distanceBetween2Points(double *coordsA, double *coordsB, int numberOfCoordsAandB){
    double result = 0.0;

    for(int i = 0; i<numberOfCoordsAandB;i++){
        result += fabs(coordsB[i] - coordsA[i]);
    }

    return result;
}

//Algo de LLoyd
/*
Parameter :
 1) Number of cluster ->  number of representant
 2) Dataset

Step :
 1) Cluster (representant)
    -> Place (or replace) cluster
 2) Each point
    -> Find nearest cluster
    -> Assing to this cluster
*/

ClusterRepresentative* algoOfLLoyd(int numberOfCluster, double *dataset, int dataset_samples_count, int dataset_sample_features_count){
    ClusterRepresentative *tabCluster = new ClusterRepresentative[numberOfCluster];
    bool stop = false;


    while(!stop) {
        //ALGO STOP check if we stop
        for (int i = 0; i < numberOfCluster; i++) {
            if (tabCluster->finalPlace) {
                stop = true;
            }
        }

        //ALGO STEP 1 : check each point in dataset
        for(int i = 0;i<dataset_samples_count;i++){
            //POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            int distMin = 1000000000;
            int theCluster = 0;
            for(int j =0; j<numberOfCluster;j++){
                //find the nearest cluster
                double *coordsA = ;
                double *coordsB = ;
                if(distanceBetween2Points(coordsA,coordsB,dataset_sample_features_count)<distMin){
                    distMin = distanceBetween2Points(coordsA,coordsB,dataset_sample_features_count);
                    theCluster = j;
                }
                //assign the inputs to the cluster
                theCluster = ;
            }

            tabCluster[theCluster].memberOfCluster[tabCluster[theCluster].countClusterMember] = dataset[i];

        }

        //ALGO STEP 2 : check cluster

    }


    return tabCluster;
}





int main() {
    std::cout << "Hello, World!" << std::endl;


    double X[2] = {2.0
                 ,2.0};


    double Y[2] = {1.0
            ,7.0};

    double  result = distanceBetween2Points(X,Y,2);
    printf("%f \n",result);
    return 0;
}
