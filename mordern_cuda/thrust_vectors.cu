#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include<iostream>

int main() {
    //initialize host vector
    thrust::host_vector<int> h_vec(10);
    for(int i = 0; i < h_vec.size(); i++) {
        h_vec[i] = rand() % 10;
    }

    //print elements in host vector
    std::cout<<"Host vector"<<" ";
    for(int i = 0; i < h_vec.size(); i++) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    //resize host vector
    h_vec.resize(5);

    //create device vector from host vector
    thrust::device_vector<int> d_vec = h_vec;
    //Modefy the last two elements in device vector 
    d_vec[3] = 10;
    d_vec[4] = 20;
    //print elements in device vector
    std::cout<<"Device vector"<<" ";
    for(int i = 0; i < d_vec.size(); i++) {
        std::cout << d_vec[i] << " ";
    }
    std::cout << std::endl;
    
    //After executions, thrust will autodelete the vectors
    return 0;
}
