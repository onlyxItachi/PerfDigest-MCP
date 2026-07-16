// template_heavy.cpp — deliberately template-heavy translation unit so that
// `clang++ -ftime-trace` has real InstantiateFunction/InstantiateClass/
// ParseTemplate/CodeGen work to report, not just an empty Frontend pass.
//
// Used ONLY to generate the clang_time_trace test fixture (see
// tests/fixtures/template_heavy.cpp + template_heavy.ftime-trace.json).

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <type_traits>
#include <memory>

// Compile-time recursion -> many InstantiateClass events.
template <unsigned N>
struct Fib {
    static constexpr unsigned long long value = Fib<N - 1>::value + Fib<N - 2>::value;
};
template <> struct Fib<0> { static constexpr unsigned long long value = 0; };
template <> struct Fib<1> { static constexpr unsigned long long value = 1; };

// Variadic template pack expansion -> InstantiateFunction events.
template <typename T>
T sum_one(T v) { return v; }

template <typename T, typename... Rest>
T sum_one(T first, Rest... rest) {
    return first + sum_one<T>(rest...);
}

// A generic container wrapper instantiated over several distinct types.
template <typename T>
class Box {
public:
    explicit Box(T v) : value_(std::move(v)) {}
    T const& get() const { return value_; }
    template <typename U>
    Box<U> map(U (*fn)(T const&)) const {
        return Box<U>(fn(value_));
    }

private:
    T value_;
};

template <typename Container>
typename Container::value_type sum_container(Container const& c) {
    typename Container::value_type acc{};
    for (auto const& v : c) acc += v;
    return acc;
}

static int double_it(int const& v) { return v * 2; }
static double to_double(int const& v) { return static_cast<double>(v); }

int main() {
    constexpr unsigned long long fib_20 = Fib<20>::value;
    constexpr unsigned long long fib_25 = Fib<25>::value;

    int a = sum_one(1, 2, 3, 4, 5, 6, 7, 8);
    double b = sum_one(1.5, 2.5, 3.5);
    long c = sum_one(1L, 2L, 3L, 4L);

    Box<int> bi(a);
    Box<int> bi2 = bi.map<int>(double_it);
    Box<double> bd = bi.map<double>(to_double);

    std::vector<int> vi{1, 2, 3, 4, 5, static_cast<int>(fib_20 % 100)};
    std::vector<double> vd{1.1, 2.2, 3.3, static_cast<double>(fib_25 % 100)};
    std::map<std::string, int> mi{{"a", 1}, {"b", 2}, {"c", 3}};

    int s1 = sum_container(vi);
    double s2 = sum_container(vd);

    std::sort(vi.begin(), vi.end());
    std::sort(vd.begin(), vd.end());

    auto up = std::make_unique<Box<int>>(s1);

    return static_cast<int>(bi2.get() + bd.get() + s1 + s2 + mi.size() + up->get()
                             + fib_20 % 2 + fib_25 % 2 + a + b + c);
}
