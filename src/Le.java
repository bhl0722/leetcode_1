import org.junit.Test;

import java.util.*;

public class Le {
    public int reverse(int x){
        int res=0;
        while(x!=0){
            if(res>Integer.MAX_VALUE/10 || res<Integer.MIN_VALUE/10){
                return 0;
            }
            int digit=x%10;
            x/=10;
            res=res*10+digit;
        }
        return res;
    }
    //
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] <= 0) {
                nums[i] = n + 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            int num = Math.abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        return n + 1;
    }
//    public String reverseWords(String s) {
//        s=s.trim();
//        List<String> strings = Arrays.asList(s.split("\\s+"));
//        Collections.reverse(strings);
//        return String.join(" ",strings);
//    }
     public String reverseWords(String s) {
        int left=0,right=s.length()-1;
        while (left<=right && s.charAt(left)==' ') left++;
        while (left<=right && s.charAt(right)==' ')right--;
        ArrayDeque<String> deque = new ArrayDeque<>();
        StringBuilder builder = new StringBuilder();
        while(left<=right){
            char c=s.charAt(left);
            if((builder.length()!=0)&& (c==' ')){
                deque.offerFirst(builder.toString());
                builder.setLength(0);
            }else if(c!=' '){
                builder.append(c);
            }
            left++;
        }
        deque.offerFirst(builder.toString());
        return String.join(" ",deque);
    }
    public int longestConsecutive(int[] nums){
        HashSet set=new HashSet<>();
        for(int num:nums){
            set.add(num);
        }
        int count=0;
        for(int num:nums){
            if(!set.contains(num-1)){
                int currentNum=num;
                int currentCount=1;
                while (set.contains(currentNum+1)){
                    currentNum++;
                    currentCount++;
                }
                count=Math.max(count,currentCount);
            }
        }
        return count;
    }

    //前序遍历
    public int[] preorderTraversal (TreeNode root) {
        List<Integer> list=new ArrayList<>();
        dfs(root,list);
        int[] roots=new int[list.size()];
        for(int i=0;i<list.size();i++){
            roots[i]=list.get(i);
        }
        return roots;
    }
    public void dfs(TreeNode root,List<Integer> list){
        if(root==null){
            return;
        }
        list.add(root.val);
        dfs(root.left,list);
        dfs(root.right,list);
    }

    //层序遍历
    public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
        ArrayList<ArrayList<Integer>> res=new ArrayList<>();
        if(root==null) return res;
        Deque<TreeNode> deque = new ArrayDeque<>();
        deque.add(root);
        while(!deque.isEmpty()){
            ArrayList<Integer> list=new ArrayList<>();
            int n=deque.size();
            for(int i=0;i<n;i++){
                TreeNode node=deque.poll();
                list.add(node.val);
                if(node.left!=null){
                    deque.add(node.left);
                }
                if(node.right!=null){
                    deque.add(node.right);
                }
            }
            res.add(list);
        }
        return res;
    }

    //按照之字形打印 [1,2,3,#,#,4,5]
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> res=new ArrayList<>();
        if(pRoot==null) return res;
        Deque<TreeNode> q=new ArrayDeque<>();
        q.add(pRoot);
        boolean flag=true;
        while(!q.isEmpty()){
            ArrayList<Integer> temp=new ArrayList<>();
            flag=!flag;
            int n=q.size();
            for(int i=0;i<n;i++){
                TreeNode node = q.poll();
                temp.add(node.val);
                if(node.left!=null) q.add(node.left);
                if(node.right!=null) q.add(node.right);
            }
            if(flag) Collections.reverse(temp);
            res.add(temp);
        }
        return res;
    }

    //二叉树的最大深度(层序遍历)
    public int maxDepth1(TreeNode root) {
        if(root==null) return 0;
        int res=0;
        Deque<TreeNode> q=new ArrayDeque<>();
        q.add(root);
        while(!q.isEmpty()){
            int n=q.size();
            for(int i=0;i<n;i++){
                TreeNode node = q.poll();
                if(node.left!=null) q.add(node.left);
                if(node.right!=null) q.add(node.right);
            }
            res++;
        }
        return res;
    }
    //二叉树的最大深度(递归)
    public int maxDepth2(TreeNode root) {
        if(root==null) return 0;
        return Math.max(maxDepth2(root.left),maxDepth2(root.right))+1;
    }

    //二叉树中和为某一值的路径（）
    public boolean hasPathSum (TreeNode root, int sum) {
        if(root==null) return false;
        if(root.left==null && root.right==null && root.val==sum) return true;
        return hasPathSum(root.left,sum-root.val)||hasPathSum(root.right,sum- root.val);
    }

    //二叉搜索树转换为双向链表
    TreeNode head=null;
    TreeNode pre=null;
    public TreeNode Convert(TreeNode pRootOfTree) {
        inorder(pRootOfTree);
        return head;
    }
    public void inorder(TreeNode root){
        if(root==null) return;
        inorder(root.left);
        if(pre==null){
            head=root;
            pre=root;
        }else{
            pre.right=root;
            root.left=pre;
            pre=root;
        }
        inorder(root.right);
    }

    //对称二叉树(递归)
    public boolean recursion(TreeNode root1,TreeNode root2){
        if(root1==null && root2==null) return true;
        if(root1==null || root2==null ||root1.val!= root2.val) return false;
        return recursion(root1.left,root2.right)&&recursion(root1.right,root2.left);
    }
    public boolean isSymmetrical(TreeNode pRoot) {
        return recursion(pRoot,pRoot);
    }

    //合并二叉树(递归前序遍历)
    public TreeNode mergeTrees (TreeNode t1, TreeNode t2) {
        if(t1==null) return t2;
        if(t2==null) return t1;
        TreeNode head=new TreeNode(t1.val+t2.val);
        head.left=mergeTrees(t1.left,t2.left);
        head.right=mergeTrees(t1.right,t2.right);
        return head;
    }

    //二叉树镜像(递归)
    public TreeNode Mirror (TreeNode pRoot) {
        if(pRoot==null) return null;
        TreeNode left=Mirror(pRoot.left);
        TreeNode right=Mirror(pRoot.right);
        pRoot.left=right;
        pRoot.right=left;
        return pRoot;
    }

    //判断是不是二叉搜索树(递归)
    public boolean isValidBST (TreeNode root) {
        return BST(root,Integer.MIN_VALUE,Integer.MAX_VALUE);
    }
    public boolean BST(TreeNode root,long left,long right){
        if(root==null) return true;
        if(root.val<=left || root.val>=right) return false;
        return BST(root.left,left,root.val) && BST(root.right,root.val,right);
    }

    //判断是不是完全二叉树(层序遍历)
    public boolean isCompleteTree (TreeNode root) {
        if(root==null) return true;
        Queue<TreeNode> q=new LinkedList<>();
        q.offer(root);
        TreeNode node;
        boolean notComplete=false;
        while(!q.isEmpty()){
            node=q.poll();
            if(node==null) {
                notComplete = true;
                continue;
            }
            if(notComplete) return false;
            q.offer(node.left);
            q.offer(node.right);
        }
        return true;
    }

    //判断是不是平衡二叉树(递归，求左右子树最大深度)
    boolean isBalanced=true;
    public boolean IsBalanced_Solution(TreeNode root) {
        TreeDepth(root);
        return isBalanced;
    }
    public int TreeDepth(TreeNode root) {
        if(root==null) return 0;
        int l=TreeDepth(root.left);
        int r=TreeDepth(root.right);
        if(Math.abs(l-r)>1){
            isBalanced=false;
        }
        return Math.max(l,r)+1;
    }

    //二叉搜索树的最近公共祖先(搜索路径)
    //求根结点到指定结点的路径
    public ArrayList<Integer> getPath(TreeNode root,int target){
        ArrayList<Integer> path = new ArrayList<>();
        TreeNode node=root;
        while(node.val!=target){
            path.add(node.val);
            if(target< node.val){
                node=node.left;
            }else node=node.right;
        }
        path.add(node.val);
        return path;
    }
    //找到最近公共祖先
    public int lowestCommonAncestor(TreeNode root, int p, int q) {
        ArrayList<Integer> path_p = getPath(root, p);
        ArrayList<Integer> path_q = getPath(root, q);
        int res=0;
        for(int i=0;i<path_p.size()&&i<path_q.size();i++){
            int x=path_p.get(i);
            int y=path_q.get(i);
            if(x==y) res=x;
            else break;
        }
        return res;
    }
    //二叉搜索树的最近公共祖先(递归) [1，2，4，5，3],[4，2，5，1，3],   [1,2,3,4,5,#,#]
    public int lowestCommonAncestor2(TreeNode root, int p, int q) {
        if(root==null) return -1;
        if((p<=root.val && q>=root.val) || (p>=root.val && q<=root.val))return root.val;
        else if((p<=root.val && q<=root.val)) return lowestCommonAncestor2(root.left,p,q);
        else return lowestCommonAncestor2(root.right,p,q);
    }

    //用两个栈实现队列(双栈)
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    public void push(int node) {
        stack1.push(node);
    }
    public int pop() {
        if(stack2.size()<=0){
            while(stack1.size()!=0){
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }

    //包含min函数的栈(双栈)
    Stack<Integer> stack3 = new Stack<Integer>();
    Stack<Integer> stack4 = new Stack<Integer>();
    public void push2(int node) {
        stack3.push(node);
        if(stack4.isEmpty() || node<=stack4.peek()){
            stack4.push(node);
        }else stack4.push(stack4.peek());
    }
    public void pop2() {
        stack3.pop();
        stack4.pop();
    }
    public int top2() {
        return stack3.peek();
    }
    public int min2() {
        return stack4.peek();
    }

    //有效括号序列(栈)
    public boolean isValid (String s) {
        // write code here
        Stack<Character> stack=new Stack<>();
        for(char c:s.toCharArray()){
            if(c=='(') stack.push(')');
            else if(c=='[') stack.push(']');
            else if(c=='{') stack.push('}');
            else if(stack.isEmpty() || c!=stack.pop())return false;
        }
        return stack.isEmpty();
    }

    //滑动窗口的最大值(双端队列)
    public ArrayList<Integer> maxInWindows(int [] num, int size) {
        ArrayList<Integer> res=new ArrayList<>();
        if(size<= num.length && size!=0){
            Deque<Integer> q=new ArrayDeque<>();
            for(int i=0;i<size;i++){
                while(!q.isEmpty() && num[i]>=num[q.peekLast()]){
                    q.pollLast();
                }
                q.addLast(i);
            }
            for(int i=size;i<num.length;i++){
                res.add(num[q.peekFirst()]);
                while(!q.isEmpty() && q.peekFirst()<(i-size+1)){
                    q.pollFirst();
                }
                while(!q.isEmpty() && num[i]>=num[q.peekLast()]){
                    q.pollLast();
                }
                q.addLast(i);
            }
            res.add(num[q.peekFirst()]);
        }
        return res;
    }

    //寻找第K大元素()
    public int findKth(int[] a, int n, int K) {
        return quickSort(a,0,n-1,K);
    }
    public int quickSort(int[] a,int left,int right,int K){
        if(left<=right){
            int p=partition(a,left,right);
            if(p==a.length-K)return a[p];
            else if(p<a.length-K)return quickSort(a,p+1,right,K);
            else return quickSort(a,left,p-1,K);
        }else return -1;
    }
    public int partition(int[] a,int left,int right){
        int key=a[left];
        while(left<right){
            while(left<right && a[right]>=key)right--;
            a[left]=a[right];
            while(left<right && a[left]<=key)left++;
            a[right]=a[left];
        }
        a[left]=key;
        return left;
    }

    //矩阵的最小路径和
    public int minPathSum (int[][] matrix) {
         int m=matrix.length,n=matrix[0].length;
         int[][] dp=new int[m+1][n+1];
         dp[0][0]=matrix[0][0];
         for(int i=1;i<m;i++) dp[i][0]=dp[i-1][0]+matrix[i][0];
         for(int j=1;j<n;j++) dp[0][j]=dp[0][j-1]+matrix[0][j];
         for(int i=1;i<m;i++){
             for(int j=1;j<n;j++){
                dp[i][j]=Math.min(dp[i-1][j],dp[i][j-1])+matrix[i][j];
             }
         }
         return dp[m-1][n-1];
    }

    //把数字翻译成字符串
    public int solve (String nums) {
        int[] dp=new int[nums.length()+1];
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=nums.length();i++){
            int num=Integer.parseInt(nums.substring(i-2,i));
            if(nums.charAt(i-1)=='0'){
                if(num==10 || num==20){
                    dp[i]=dp[i-1];
                    continue;
                }else return 0;
            }else if(num>10 && num<=26){
                dp[i]=dp[i-1]+dp[i-2];
            }else dp[i]=dp[i-1];
        }
        return dp[nums.length()];
    }

    //兑换零钱
    //[1,2,3,4,5,100],0
    public int minMoney (int[] arr, int aim) {
        if(aim<1) return 0;
        int[] dp=new int[aim+1];
        Arrays.fill(dp,aim+1);
        dp[0]=0;
        for(int i=1;i<=aim;i++){
            for(int j=0;j<arr.length;j++){
                if(arr[j]<=i){
                    dp[i]=Math.min(dp[i],dp[i-arr[j]]+1);
                }
            }
        }
        return dp[aim]>aim ?-1:dp[aim];
    }

    //最长上升子序列  [6,3,1,5,2,3,7]
    public int LIS (int[] arr) {
        int res=0;
        int[] dp=new int[arr.length];
        Arrays.fill(dp,1);
        for(int i=1;i<arr.length;i++){
            for(int j=0;j<i;j++){
                if(arr[i]>arr[j] && dp[i]<dp[j]+1){
                    dp[i]=dp[j]+1;
                    res=Math.max(res,dp[i]);
                }
            }
        }
        return res;
    }

    //连续子数组的最大和   [1,-2,3,10,-4,7,2,-5] 18
    public int FindGreatestSumOfSubArray(int[] array) {
        int[] dp=new int[array.length];
        int max=array[0];
        dp[0]= array[0];
        for(int i=1;i< array.length;i++){
            dp[i]=Math.max(dp[i-1]+array[i],array[i]);
            max=Math.max(max,dp[i]);
        }
        return max;
    }

    //最长回文子串  ababc  3
    public int getLongestPalindrome (String A) {
        int n=A.length();
        boolean[][] dp=new boolean[n][n];
        int max=0;
        for(int c=0;c<=n+1;c++){
            for(int i=0;i<n-c;i++){
                int j=i+c;
                if(A.charAt(i)==A.charAt(j)){
                    if(c<=1) dp[i][j]=true;
                    else dp[i][j]=dp[i+1][j-1];
                }
                if(dp[i][j]) max=c+1;
            }
        }
        return max;
    }

    //编辑距离一 "nowcoder","new" 6
    public int editDistance (String str1, String str2) {
        int m=str1.length(),n=str2.length();
        int[][] dp=new int[m+1][n+1];
        for(int i=1;i<=m;i++) {
            dp[i][0]=dp[i-1][0]+1;
        }
        for(int j=1;j<=n;j++){
            dp[0][j]=dp[0][j-1]+1;
        }
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(str1.charAt(i-1)==str2.charAt(j-1))
                    dp[i][j]=dp[i-1][j-1];
                else
                    dp[i][j]=Math.min(dp[i-1][j-1],Math.min(dp[i-1][j],dp[i][j-1]))+1;
            }
        }
        return dp[m][n];
    }

    //正则表达式匹配   "aaa","a*a"  true
    public boolean match (String str, String pattern) {
        int m=str.length(),n=pattern.length();
        boolean[][] dp=new boolean[m+1][n+1];
        dp[0][0]=true;
        for(int i=0;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(pattern.charAt(j-1)!='*'){
                    if(i>0 && str.charAt(i-1)==pattern.charAt(j-1) || pattern.charAt(j-1)=='.')
                            dp[i][j]=dp[i-1][j-1];
                }else{
                    if(j>=2)
                        dp[i][j]=dp[i][j-2];
                    if(i>=1 && j>=2 && (str.charAt(i-1)==pattern.charAt(j-2) || pattern.charAt(j-2)=='.'))
                        dp[i][j]=dp[i-1][j] || dp[i][j-2];
                }
            }
        }
        return dp[m][n];
    }

    //打家结舍1  [1,2,3,4]->6
    public int rob (int[] nums) {
        int[] dp=new int[nums.length+1];
        dp[1]=nums[0];
        for(int i=2;i<=nums.length;i++){
            dp[i]=Math.max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        return dp[nums.length];
    }

    //打家劫舍2  [1,3,6] （第一家和最后一家不能同时偷）->6
    public int rob2(int[] nums) {
        int[] dp=new int[nums.length+1];
        dp[1]=nums[0];
        for(int i=2;i<nums.length;i++){
            dp[i]=Math.max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        int res=dp[nums.length-1];
        Arrays.fill(dp,0);
        for(int i=2;i<=nums.length;i++){
            dp[i]=Math.max(dp[i-1],dp[i-2]+nums[i-1]);
        }
        return Math.max(res,dp[nums.length]);
    }

    //买卖股票的最佳时机1(只能买入卖出一次)   [8,9,2,5,4,7,1]-> 5
    public int maxProfit(int[] prices) {
//        int[][] dp=new int[prices.length][2];
//        dp[0][0]=0;
//        dp[0][1]=-prices[0];
//        for(int i=1;i< prices.length;i++){
//            dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
//            dp[i][1]=Math.max(dp[i-1][1],-prices[i]);
//        }
//        return dp[prices.length-1][0];
        int minPrice=Integer.MAX_VALUE;
        int res=0;
        for(int i=0;i<prices.length;i++){
            res=Math.max(res,prices[i]-minPrice);
            minPrice=Math.min(minPrice,prices[i]);
        }
        return res;
    }

    //买卖股票的最佳时机2(可以多次买入卖出)   [8,9,2,5,4,7,1]-> 7
    public int maxProfit2(int[] prices) {
//        int[][] dp=new int[prices.length][2];
//        dp[0][0]=0;
//        dp[0][1]=-prices[0];
//        for(int i=1;i<prices.length;i++){
//            dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
//            dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i]);
//        }
//        return dp[prices.length-1][0];
        int res=0;
        for(int i=1;i<prices.length;i++){
            if(prices[i]>prices[i-1]){
                res+=prices[i]-prices[i-1];
            }
        }
        return res;
    }

    //买卖股票的最佳时机3(最多买入卖出2次，且第二次买之前必须卖出一次)   [8,9,3,5,1,3]-> 4
    public int maxProfit3(int[] prices) {
        int[][] dp=new int[prices.length][4];
        Arrays.fill(dp[0],-10000);
        
        return 0;
    }
}
