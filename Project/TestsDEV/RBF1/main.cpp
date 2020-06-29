#include <iostream>
#include <random>

struct ClusterRepresentative{
    double *coord;
    double *datasetInputs;
    bool finalPlace;

    void initialise(int,int);
};


void ClusterRepresentative::initialise(int dataset_samples_count,int dataset_sample_features_count){

    coord = new double[dataset_sample_features_count];

    for(int i = 0; i<dataset_sample_features_count;i++) {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist(0, 20);
        auto randNumber = dist(mt);
        coord[i] = randNumber;
    }


    datasetInputs = new double[dataset_samples_count];
    finalPlace = false;
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

ClusterRepresentative* algoOfLLoyd(int numberOfCluster, double *dataset){
    ClusterRepresentative *tabCluster = new ClusterRepresentative[numberOfCluster];
    bool stop = false;


    while(!stop) {
        //check if we stop
        for (int i = 0; i < numberOfCluster; i++) {
            if (tabCluster->finalPlace) {
                stop = true;
            }
        }



    }


    return tabCluster;
}



int main() {
    std::cout << "Hello, World!" << std::endl;


    return 0;
}
