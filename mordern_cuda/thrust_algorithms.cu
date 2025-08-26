#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/universal_vector.h>
#include<iostream>
//1. Transformation
#include<thrust/functional.h>
#include<thrust/sequence.h>
#include<thrust/copy.h>
#include<thrust/replace.h>


void transformation() {
    //define a device vector
    thrust::device_vector<int> d_X(10);
    thrust::device_vector<int> d_Y(10);
    thrust::device_vector<int> d_Z(10);
    //initialize X from 1 to 10
    thrust::sequence(d_X.begin(), d_X.end());
    //print X
    std::cout<<"\nX: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //transfrorm Y = -X
    thrust::transform(d_X.begin(), d_X.end(), d_Y.begin(), thrust::negate<int>());
    //print Y
    std::cout<<"\nY = -X: ";
    for(auto it = d_Y.begin(); it != d_Y.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;
    
    //fill Z with 2s
    thrust::fill(d_Z.begin(), d_Z.end(), 2);
    
    //Compute Y = X mod Z
    thrust::transform(d_X.begin(), d_X.end(), d_Z.begin(), d_Y.begin(), thrust::modulus<int>());
    //print Y
    std::cout<<"\nY = X mod Z: ";
    for(auto it = d_Y.begin(); it != d_Y.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;
    
    //Replace all 1 s in Y with 10s
    thrust::replace(d_Y.begin(), d_Y.end(), 1, 10);
    //print Y
    std::cout<<"\nY = X mod Z with all 1s replaced with 10s: "<<std::endl;
    for(auto it = d_Y.begin(); it != d_Y.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //Print Y by copying Y to std::stream operator
    std::cout<<"\nY = X mod Z with all 1s replaced with 10s (Stream operator print): "<<std::endl;
    thrust::copy(d_Y.begin(), d_Y.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout<<std::endl;
}

//2. Reduction
void reduction() {
    //Define a device vector from 1 to 10
    thrust::device_vector<int> d_X(10);
    thrust::sequence(d_X.begin(), d_X.end(), 1);
    //print X
    std::cout<<"\nX: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;
    
    //reduce sum of X
    std::cout<<"\nSum of X: "<<thrust::reduce(d_X.begin(), d_X.end())<<std::endl;
}

//3. Prefix-sum
#include<thrust/scan.h>
void prefixSum() {
    //Define a device vector and fill it with constant value
    thrust::device_vector<int> d_X(10);
    thrust::fill(d_X.begin(), d_X.end(), 3);
    std::cout<<"\nX: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //inplace scan
    thrust::inclusive_scan(d_X.begin(), d_X.begin() + 7, d_X.begin());//3 6 9 12 15 18 21 3 3 3 Initial value not possible
    
    //print X
    std::cout<<"\nX after innclusive scan: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //replace x with 1 again
    thrust::fill(d_X.begin(), d_X.end(), 3);

    //prefix_sum
    //thrust::exclusive_scan(d_X.begin(), d_X.begin()+7, d_X.begin());//0 3 6 9 12 15 18 3 3 3 
    thrust::exclusive_scan(d_X.begin(), d_X.begin()+7, d_X.begin(),10);//Initial value 10
    //print X
    std::cout<<"\nX after exclusive scan: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;
 
}

//4. Re-ordering
#include<thrust/partition.h>
#include<thrust/unique.h>
void reordering(){
    //create a device vector with serial values
    thrust::device_vector<int> d_X(10);
    for(int i = 0; i < d_X.size(); i++) {
        d_X[i] = rand() % 10;
    }
    //print X
    std::cout<<"\nX: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //filter X using partition iterator
    auto partition = thrust::partition(d_X.begin(), d_X.end(), [] __host__ __device__(int x) { return x % 2 == 0; });
    //print X
    std::cout<<"\nX after partition filter: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //create an output vector
    thrust::device_vector<int> d_Y(d_X.size());
    thrust::fill(d_Y.begin(), d_Y.end(), 0);
    //filter X using copy_if
    auto filtered = thrust::copy_if(d_X.begin(), d_X.end(), d_Y.begin(), [] __host__ __device__(int x) { return x % 2 == 0; });
    //print Y
    std::cout<<"\nY after copy_if: ";
    for(auto it = d_Y.begin(); it != d_Y.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //remove duplicate from Y using unique
    auto unique = thrust::unique(d_Y.begin(), d_Y.end());
    //print Y
    std::cout<<"\nY after removing duplicate using unique: ";
    for(auto it = d_Y.begin(); it != d_Y.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //remove_if
    auto removed = thrust::remove_if(d_X.begin(), d_X.end(), [] __host__ __device__(int x) { return x % 2 == 0; });
    //print X
    std::cout<<"\nX after remove_if: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;
}

//5. Sorting
#include<thrust/sort.h>
void sorting(){
    //create a device vector with serial values
    thrust::device_vector<int> d_X(10);
    for(int i = 0; i < d_X.size(); i++) {
        d_X[i] = rand() % 10;
    }
    //print X
    std::cout<<"\nX: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //sort X ascending
    thrust::sort(d_X.begin(), d_X.end());
    //print X
    std::cout<<"\nX after ascending sort: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;

    //short descending
    thrust::sort(d_X.begin(), d_X.end(), thrust::greater<int>());
    //print X
    std::cout<<"\nX after descending sort: ";
    for(auto it = d_X.begin(); it != d_X.end(); it++) { std::cout<<*it<<" "; }
    std::cout<<std::endl;
}


//Inumerator the numbers are commandline arguments
enum {
    TRANSFORMATION = 1,
    REDUCTION = 2,
    PREFIX_SUM = 3,
    REORDERING = 4,
    SORTING = 5,
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <function_number> needed" << std::endl;
        return 1;
    }
    int functionmatch = std::stoi(argv[1]);
    
    switch(functionmatch) {
        case TRANSFORMATION:
            transformation();
            break;
        case REDUCTION:
            reduction();
            break;
        case PREFIX_SUM:
            prefixSum();
            break;
        case REORDERING:
            reordering();
            break;
        case SORTING:
            sorting();
            break;
        default:
            std::cout << "Invalid function name" << std::endl;
            return 1;
    }
    return 0;
}