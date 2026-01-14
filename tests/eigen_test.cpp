#include <Eigen/Core>

int func(const Eigen::Matrix<int, 4, 2>& p)
{
    Eigen::Matrix<int, 4, 2> l_mat = p;
}

int main()
{
    int a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Eigen::Map<Eigen::Matrix<int, 4,2>> map_a(a);
    func(map_a);
    return 0;
}
