class Solution {
public:
    string longestPalindrome(string s) {
        // 用vector的原因是因为数组的大小在编译时必须是常量表达式 int a[s.size()][s.size()]是非法的
        vector<vector<int>> dp(s.size(), vector<int>(s.size(), false));
        int result = 0;
        string str;
        // 从后往前找，相同回文数长度越前面的优先级越高，覆盖后面的
        for (int i = s.size() - 1; i >= 0; i--) {
            for (int j = i; j < s.size(); j++) {
                // j-i <= 1:单个字符
                // dp[i][j-1]:子回文数
                if (s[i] == s[j] && (j - i <= 1 || dp[i + 1][j - 1])) {
                    dp[i][j] = true;
                    result = max(result, j - i);
                    if (j - i == result) str = s.substr(i, j - i + 1);
                }
            }
        }
        return str;
        
    }
};
