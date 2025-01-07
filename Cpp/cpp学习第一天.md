## c++学习第一天(指针和引用)

## POINTER in C++

- C++中的指针就是一个**整数**，它存储一个电脑中的内存地址，内存地址所对应的可以是一个数据（指针），也可以是一个地址，该地址所再内存又存储一个数据或者内存地址（多级指针）

```cpp
#include <iostream>

#define LOG(X) std::cout << X << std::endl;

int main()
{
	int var = 16;
	int* ptr = &var;
	*ptr = 10;
	int** pptr = &ptr;
	LOG(**pptr);
	std::cin.get();
    
	return 0;
}
```

- 通过此例程，可以理解指针以及二级指针。再VS2022上对程序打断点进行调试，并观察内存窗口的数据，可以很好的理解指针的工作原理。

## REFERENCE in C++

引用变量是一个别名，也就是说，它是某个已存在变量的另一个名字。一旦把引用初始化为某个变量，就可以使用该引用名称或变量名称来指向变量。

- 一旦引用被初始化为一个对象，就不能被指向到另一个对象。指针可以在任何时候指向到另一个对象。

- 引用的定义方法：

  ```cpp
  int var = 1;
  int& ref = var;	//the value of ref = the value og var
  ```

- 相比于指针，引用可以使函数更易读懂更简洁，例如实现一个自加功能的函数：

  使用指针方法：

  ```cpp
  #include <iostream>
  
  #define LOG(X) std::cout << X << std::endl;
  
  void Increment(int* value)
  {
  	(*value)++;	//加括号的原因是运算符的优先级，++>*
  }
  
  int main()
  {
  	int var = 1;
  	Increment(&var);
  	LOG(ref);
  	return 0;
  }
  ```

  而引用就可以写为：

  ```cpp
  #include <iostream>
  
  #define LOG(X) std::cout << X << std::endl;
  
  void Increment(int& value)
  {
  	value++;
  }
  
  int main()
  {
  	int var = 1;
  	Increment(var);
  	LOG(ref);
  	return 0;
  }
  ```

## CLASS in C++

默认私有

```cpp
class player {
 int x, y;
 int speed;
 void move(int a, int b){
     x += a * speed;
     y += b * speed;
 }
};
```

访问控制（private、public）

```cpp
class player {
public:
 int x, y;
 int speed;
private:
 void move(int a, int b){
     x += a * speed;
     y += b * speed;
 }
};
```

### CLASSES vs STRUCTS

- structs默认是public二classes默认私有，c++保留struct是为了与c兼容
- 通常使用struct描述数据结构，比如描述数学中的向量更多用struct，struct不会使用继承，继承增加了一种复杂性