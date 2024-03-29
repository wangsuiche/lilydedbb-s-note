## [【hihocoder】1043 : 完全背包](https://hihocoder.com/problemset/problem/1043)

时间限制:20000ms

单点时限:1000ms

内存限制:256MB

### 输入

每个测试点（输入文件）有且仅有一组测试数据。

每组测试数据的第一行为两个正整数N和M,表示奖品的种数，以及小Ho手中的奖券数。

接下来的n行描述每一行描述一种奖品，其中第i行为两个整数need(i)和value(i)，意义如前文所述。

测试数据保证

对于100%的数据，N的值不超过500，M的值不超过10^5

对于100%的数据，need(i)不超过2*10^5, value(i)不超过10^3

### 输出

对于每组测试数据，输出一个整数Ans，表示小Ho可以获得的总喜好值。

#### 样例输入

```
5 1000
144 990
487 436
210 673
567 58
1056 897
```

#### 样例输出

```
5940
```

```c
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Obj {
    int need, value;
    Obj(int n, int v): need(n), value(v) {};
};

int main(int argc, const char * argv[]) {

    int n, m, need, value;
    cin >> n >> m;
    vector<Obj*> objs;
    for (int i = 0; i < n; i++) {
        cin >> need >> value;
        objs.push_back(new Obj(need, value));
    }
    vector<vector<int>> dp(n, vector<int>(m + 1, 0));
    for (int i = 0; i < n; i++) {
        int w = objs[i]->need, v = objs[i]->value;
        for (int j = 0; j <= m; j++) {
            if (w > j)
                dp[i][j] = (i > 0) ? dp[i - 1][j] : 0;
            else
                dp[i][j] = max((i > 0) ? dp[i - 1][j] : 0, dp[i][j - w] + v);
        }
    }
    cout << dp[n - 1][m] << endl;
    return 0;
}
```