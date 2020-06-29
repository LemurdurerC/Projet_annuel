#include <iostream>
#include <random>

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
        std::uniform_int_distribution<int> dist(0, 20);
        auto randNumber = dist(mt);
        coord[i] = randNumber;
        formerCoord[i] = coord[i];
    }


    *memberOfCluster = new double[dataset_samples_count/dataset_sample_features_count];
    for(int i = 0; i<dataset_samples_count/dataset_sample_features_count;i++){
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

        //ALGO STEP 1 : check each point in dataset
        for(int i = 0;i<dataset_samples_count;i=i+dataset_sample_features_count){
            //POTENTIAL ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            int distMin = 1000000000;
            int theCluster = 0;
            for(int j =0; j<numberOfCluster;j++){
                //find the nearest cluster
                //just for C++ test
                double *coordsA = getPartsOfTab(i,i+dataset_sample_features_count-1,dataset);
                double *coordsB = tabCluster[j].coord;
                if(distanceBetween2Points(coordsA,coordsB,dataset_sample_features_count)<distMin){
                    distMin = distanceBetween2Points(coordsA,coordsB,dataset_sample_features_count);
                    theCluster = j;
                }
            }

            //assign the inputs to the right cluster
            double *dataInCluster = getPartsOfTab(i,i+dataset_sample_features_count-1,dataset);

            tabCluster[theCluster].memberOfCluster[tabCluster[theCluster].countClusterMember] = dataInCluster;
            tabCluster[theCluster].countClusterMember++;

        }


        //ALGO STEP 2 : check cluster
        for(int i =0; i<numberOfCluster;i++) {
            copyCoordAtoCoordB(tabCluster[i].coord, tabCluster[i].formerCoord,dataset_sample_features_count);

            //mean of all point assigned to cluster i
            double *newCoord = new double[dataset_sample_features_count];
            initialiseTabTo0(newCoord,dataset_sample_features_count);

            for(int j = 0;j<tabCluster[i].countClusterMember;j++){
                //sum of all coord of point in cluster i
                for(int k = 0; k<dataset_sample_features_count;k++){
                    newCoord[k] = newCoord[k] + tabCluster[i].memberOfCluster[j][k];
                }
            }

            for(int k = 0; k<dataset_sample_features_count;k++){
                newCoord[k] = newCoord[k] / tabCluster[i].countClusterMember;
            }

            copyCoordAtoCoordB(newCoord, tabCluster[i].coord,dataset_sample_features_count);

            if(checkEquality2Points(tabCluster[i].formerCoord,tabCluster[i].coord, dataset_sample_features_count)){
                tabCluster[i].finalPlace = true;
            }

        }




        //ALGO STOP check if we stop
        int finishCluster = 0;
        for (int i = 0; i < numberOfCluster; i++) {
            if (tabCluster[i].finalPlace) {
                finishCluster++;
            }
        }
        if(finishCluster == numberOfCluster){
            stop = true;
        }



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
