# GIT指南

参考教程[Github如何上传项目(超详细小白教程)_github上传项目-CSDN博客](https://blog.csdn.net/KevinRay_0854/article/details/140408003?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-140408003-blog-86142418.235^v43^pc_blog_bottom_relevance_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~YuanLiJiHua~Position-3-140408003-blog-86142418.235^v43^pc_blog_bottom_relevance_base3)

## 创建仓库（GitHub）

在github主页上建立一个仓库

仓库注释：

<img src="https://i-blog.csdnimg.cn/direct/32f66cf6550e44ada7067e89d6c02175.png" alt="仓库初始化填写" style="zoom: 33%;" />

## 本地部署（gitbash）

1. 下载gitbash

2. 建立一个存放仓库的文件夹

3. 在该文件夹的目录下启动gitbash

4. 在gitbash中输入`git init`

5. 此时在该文件夹中会生成.git文件夹，但由于.git文件夹是隐藏的，可以使用`ls -a`来查看该文件夹

   ![image-20240831171756949](GIT%E6%8C%87%E5%8D%97.assets/image-20240831171756949.png)

6. 进入.git目录后，使用vim对config进行编辑，写入如下：

   ![image-20240831171934999](GIT%E6%8C%87%E5%8D%97.assets/image-20240831171934999.png)

7. 上两步也可以替换为在仓库的目录下使用如下命令进行连接

   ```bash
   git config --global user.email "youraddress@company.com"
   git config --global user.name "yourname"		
   ```

8. 链接远端仓库

   ```bash
   git remote add origin 你的URL
   ```

9. 切换分支或提交代码

## GitHub的上传原理

![Github上传原理图](https://i-blog.csdnimg.cn/direct/21774dabe40b437cb0907ca385dbe2ed.png#pic_center)

如图所示，是Github上传的一个原理图，我们的电脑就是workspace，当我们执行add和commit命令后，项目文件会被推送到一个中间仓库，它既不在本地也不在Github远端仓库，可以用于临时保存文件。然后使用push命令，将文件推送到Github仓库管理，这时文件将被Github保存起来，可以随时拉取文件，所以我们主要做三步：**第一，将项目文件加到缓冲区；第二，将文件提交到中间仓库；第三，将文件推送至Github。但是在这之前需要对本地仓库进行配置。**

## git协同办公

最近在笔记本和工作电脑上配置了git想着协同办公，将GitHub的仓库变成一个共享文件夹。但这其中有个问题困扰了我，就是在`git pull`和`git push`命令对本地和github仓库文件同步的问题。

现在设想

1. 笔记本电脑更改了某一文件，将其`git push`到github上，发现还没有与云端仓库进行merge?那这时使用`git pull`会不会覆盖掉我更改的文件，还是比较智能的覆盖？
2. 

这里我不太清楚git的工作流程，需要学习一下。
