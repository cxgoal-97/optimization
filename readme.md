## 项目说明

本仓库是2019Fall最优化作业代码。主要包含：

+ 求步长方法
  + 精确求步长方法:0.618法
  + 非精确求步长方法:
    + 单调方法：下降准则：Gold-Stein，Wolfe，Wolfe-Power。求步长：插值方法，2点2次，3点3次；迭代方法：backtraing
    + 非单调方法：GLL方法
+ 求下降方向的方法
  + L-BFGS方法:包括常规形式和压缩形式的BFGS方法
  + 牛顿型方法：牛顿法，修正牛顿法(GillMurrayNewton方法)，FletcherFreemanMethod
  + 信赖域方法: 求解子问题方法包括：二维子空间法，柯西点法，Hebden方法。
  + 非精确牛顿方法
  + 最速下降法



希望能帮助到有需要的同学。



需要的类：numpy, scipy, math, decimal



## 使用说明

1. init a suitable step optimizer 

   初始化求步长的优化器（如果需要的话）

2. init a suitable method optimizer 

   初始化具体的优化器

3. use the method_optimizer.compute to compute the minimum value of function

   调用method_optimizer.compute 方法求最优化问题

   

## 目录结构说明

|--- readme.md //说明

|--- linearSearch //线搜索方法集合

|	|--- linearSearch.py // 线搜索模板类

|	|--- monotoneSearch.py 

|    |--- nonmonotoneSearch.py 

|--- optimizer // 最优化方法集合

|	|--- basicOptimizer.py // 优化模板类

|	|--- bfgs.py 

|	|--- inaccurateNewton.py

|	|--- newtonMethod.py

|	|--- steepestDescent.py

|	|--- trustRegionMethod.py

|--- utils //工具函数集合

|	|--- distance.py

|	|--- matrixEquation.py

|	|--- matrixFraction.py

|	|--- nrom.py

|--- testFunction.py //测试函数集合

|--- 测试用例.ipynb