#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include<thrust/copy.h>
#include<list>
#include<iostream>

//1. stl to thrust device vector copy
void stlToThrustCopy() {
    //create a list with four values
    std::list<int> stl_list; //= {1, 2, 3, 4};
    stl_list.push_back(1);
    stl_list.push_back(2);
    stl_list.push_back(3);
    stl_list.push_back(4);
    
    //create a device vector and initialize it with the values from the list
    thrust::device_vector<int> d_vec(stl_list.begin(), stl_list.end());
    
    //copy the values from the device vector to the host vector
    thrust::copy(d_vec.begin(), d_vec.end(), stl_list.begin());
    //print the values in the host vector
    std::cout<<"Host vector"<<" ";
    for(auto it = stl_list.begin(); it != stl_list.end(); it++) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

//2. Constant iterator 
#include<thrust/iterator/counting_iterator.h>
void constantIterator() {
    //create iterators
    thrust::counting_iterator<int> first(10);//Constant iterator with value 10
    thrust::counting_iterator<int> last = first + 3;//last is not included

    std::cout<<"first[0]: "<<first[0]<<std::endl;
    std::cout<<"first[10]: "<<first[10]<<std::endl;
    std::cout<<"first[100]: "<<first[100]<<std::endl;
    
    //Reduce sum of the constant iterator
    std::cout<<"thrust::reduce(first, last): "<<thrust::reduce(first, last)<<std::endl;
}

//3. transforms
#include<thrust/iterator/transform_iterator.h>
#include<thrust/functional.h>
//User defined unary function syntax
/*
struct my_unary_function{
    __host__ __device__
    int operator()(const int& x) const {
        return x * x + 1;  // Example transformation
    }
};
*/

void transformIterator() {
    //create a device vector
    thrust::device_vector<int> d_vec(3);
    d_vec[0] = 10;
    d_vec[1] = 20;
    d_vec[2] = 30;

    auto first = thrust::make_transform_iterator(d_vec.begin(), thrust::negate<int>());
    auto last = thrust::make_transform_iterator(d_vec.end(), thrust::negate<int>());
    std::cout<<"first[1]: "<<first[0]<<std::endl;//first[1] is the first element of the transformed vector
    std::cout<<"first[2]: "<<first[1]<<std::endl;//first[2] is the second element of the transformed vector
    std::cout<<"first[3]: "<<first[2]<<std::endl;//first[3] is the third element of the transformed vector
    
    //Reduce, sum of the transformed vector
    std::cout<<"thrust::reduce(first, last): "<<thrust::reduce(first, last)<<std::endl;
}

//4. Permutation iterators
#include<thrust/iterator/permutation_iterator.h>
void permutationIterator() {
    //location on a device vector
    thrust::device_vector<int> d_map(10);
    for(int i = 0; i < d_map.size(); i++) {
        d_map[i] = rand() % 10;
        std::cout<<d_map[i]<<" ";
    }
    std::cout<<std::endl;
    //Array to gather from
    thrust::device_vector<int> d_gather(100);
    for(int i = 0; i < d_gather.size(); i++) {
        d_gather[i] = rand() % 10;
    }
    
    //permutation iterator
    auto p_a = thrust::make_permutation_iterator(d_gather.begin(), d_map.begin());
    auto p_b = thrust::make_permutation_iterator(d_gather.begin(), d_map.end());

    //print the permutation iterator
    for(auto it = p_a; it != p_b; it++) {
        std::cout<<"d_gather["<<*it<<"] = "<<d_gather[*it]<<std::endl;
    }
    std::cout<<std::endl;
    
    //reduce sum of the permutation iterator
    std::cout<<"thrust::reduce(p_a, p_b): "<<thrust::reduce(p_a, p_b)<<std::endl;  
}


//5. Zip iterator
#include<thrust/iterator/zip_iterator.h>
void zipIterator() {
    //A and b device vectors
    thrust::device_vector<int> d_a(10);
    thrust::device_vector<int> d_b(10);
    std::cout<<"Creating vectors"<<std::endl;
    for(int i = 0; i < d_a.size(); i++) {
        d_a[i] = rand() % 10;
        d_b[i] = rand() % 10;
        std::cout<<d_a[i]<<","<<d_b[i]<<"; ";
    }
    std::cout<<std::endl;
    
    auto first = thrust::make_zip_iterator(thrust::make_tuple(d_a.begin(), d_b.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(d_a.end(), d_b.end()));
    
    //print the zip iterator
    std::cout<<"Zipped vectors"<<std::endl;
    for(auto it = first; it != last; it++) {
        std::cout<<"("<<thrust::get<0>(*it)<<","<<thrust::get<1>(*it)<<")"<<"   ";
    }
    std::cout<<std::endl;
}

//enum for the cases follow the numbers for command line arguments
enum {
    STL_TO_THRUST_COPY = 1,
    CONSTANT_ITERATOR = 2,
    TRANSFORM_ITERATOR = 3,
    PERMUTATION_ITERATOR = 4,
    ZIP_ITERATOR = 5,
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <function_number> needed" << std::endl;
        return 1;
    }
    int functionmatch = std::stoi(argv[1]);
    
    switch(functionmatch) {
        case STL_TO_THRUST_COPY:
            stlToThrustCopy();
            break;
        case CONSTANT_ITERATOR:
            constantIterator();
            break;
        case TRANSFORM_ITERATOR:
            transformIterator();
            break;
        case PERMUTATION_ITERATOR:
            permutationIterator();
            break;
        case ZIP_ITERATOR:
            zipIterator();
            break;
        default:
            std::cout << "Invalid function name" << std::endl;
            return 1;
    }
    return 0;
}
