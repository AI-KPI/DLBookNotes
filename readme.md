<p align="center">
    <a href="https://www.deeplearningbook.org/" title="deeplearningbook">
<img src="deep-learning-book-goodfellow-cover.jpg" width="400" alt="Cover of the deep learning book by Goodfellow, Bengio and Courville" title="The Deep Learning Book - Goodfellow, I., Bengio, Y., and Courville, A. (2016)">
</a>
</p>

# Boost your data science skills. Learn linear algebra

**The Deep Learning Book - Goodfellow, I., Bengio, Y., and Courville, A. (2016)**

This content is part of a series following the chapter 2 on linear algebra from the [Deep Learning Book](http://www.deeplearningbook.org/) by Goodfellow, I., Bengio, Y., and Courville, A. (2016). The aim of these notebooks is to help beginners/advanced beginners to grasp linear algebra concepts underlying deep learning and machine learning. Acquiring these skills can boost your ability to understand and apply various machine learning, deep learning and data science algorithms. You can find additional articles [here](https://hadrienj.github.io).



# Syllabus

The syllabus follows exactly the [Deep Learning Book](http://www.deeplearningbook.org/) so you can find more details if you can't understand one specific point while you are reading it. Here is a short description of the content:

1. [2.1 Scalars, Vectors, Matrices and Tensors](https://github.com/AI-KPI/DLBookNotes/blob/main/2.1%20Scalars%2C%20Vectors%2C%20Matrices%20and%20Tensors/2_1_Scalars_Vectors_Matrices_and_Tensors.ipynb)

    <img src="2.1 Scalars, Vectors, Matrices and Tensors/images/scalar-vector-matrix-tensor.png" width="400" alt="An example of a scalar, a vector, a matrix and a tensor" title="Difference between a scalar, a vector, a matrix and a tensor">

    **Difference between a scalar, a vector, a matrix and a tensor**

    Light introduction to vectors, matrices, transpose and basic operations (addition of vectors of matrices). Introduces also Numpy functions and finally a word on broadcasting.

2. [Multiplying Matrices and Vectors](https://github.com/AI-KPI/DLBookNotes/blob/main/2.2%20Multiplying%20Matrices%20and%20Vectors/2.2%20Multiplying%20Matrices%20and%20Vectors.ipynb)

    <img src="2.2 Multiplying Matrices and Vectors/images/dot-product.png" width="400" alt="An example of how to calculate the dot product" title="The dot product explained">

    **The dot product explained**

    This chapter is mainly on the dot product (vector and/or matrix multiplication). We will also see some of its properties. Then, we will see how to synthesize a system of linear equations using matrix notation. This is a major process for the following chapters.

3. [Identity and Inverse Matrices](https://github.com/AI-KPI/DLBookNotes/blob/main/2.3%20Identity%20and%20Inverse%20Matrices/2.3_Identity_and_Inverse_Matrices.ipynb)

    <img src="2.3 Identity and Inverse Matrices/images/identity-matrix.png" width="200" alt="Example of an identity matrix" title="An identity matrix">

    **An identity matrix**

    We will see two important matrices: the identity matrix and the inverse matrix. We will see why they are important in linear algebra and how to use them with Numpy. Finally, we will see an example on how to solve a system of linear equations with the inverse matrix.

4. [Linear Dependence and Span](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.4-Linear-Dependence-and-Span/)

    <img src="2.4 Linear Dependence and Span/images/number-solutions-system-equations.png" width="700" alt="Examples of systems of equations with 0, 1 and an infinite number of solutions" title="System of equations with 0, 1 and an infinite number of solutions">

    **A system of equations has no solution, 1 solution or an infinite number of solutions**

    In this chapter we will continue to study systems of linear equations. We will see that such systems can't have more than one solution and less than an infinite number of solutions. We will see the intuition, the graphical representation and the proof behind this statement. Then we will go back to the matrix form of the system and consider what Gilbert Strang calls the *row figure* (we are looking at the rows, that is to say multiple equations) and the *column figure* (looking at the columns, that is to say the linear combination of the coefficients). We will also see what is linear combination. Finally, we will see examples of overdetermined and underdetermined systems of equations.

5. [Norms](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.5-Norms/)

    <img src="2.5 Norms/images/squared-L2-Norm.png" width="400" alt="Representation of the squared L2 norm in 3 dimensions" title="Representation of the squared L2 norm in 3 dimensions">

    **Shape of a squared L2 norm in 3 dimensions**

    The norm of a vector is a function that takes a vector in input and outputs a positive value. It can be thought of as the *length* of the vector. It is for example used to evaluate the distance between the prediction of a model and the actual value. We will see different kinds of norms ($L^0$, $L^1$, $L^2$...) with examples.

6. [Special Kinds of Matrices and Vectors](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.6-Special-Kinds-of-Matrices-and-Vectors/)

    <img src="2.6 Special Kinds of Matrices and Vectors/images/diagonal-and-symmetric-matrices.png" width="400" alt="Example of a diagonal matrix and of a symmetric matrix" title="Example of a diagonal matrix and of a symmetric matrix">

    **A diagonal (left) and a symmetric matrix (right)**

    We have seen in [2.3](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.3-Identity-and-Inverse-Matrices/) some special matrices that are very interesting. We will see other types of vectors and matrices in this chapter. It is not a big chapter but it is important to understand the next ones.

7. [Eigendecomposition](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.7-Eigendecomposition/)

    <img src="2.7 Eigendecomposition/images/output_59_0.png" width="300" alt="output_59_0">

    We will see some major concepts of linear algebra in this chapter. We will start by getting some ideas on eigenvectors and eigenvalues. We will see that a matrix can be seen as a linear transformation and that applying a matrix on its eigenvectors gives new vectors with same direction. Then we will see how to express quadratic equations in a matrix form. We will see that the eigendecomposition of the matrix corresponding to the quadratic equation can be used to find its minimum and maximum. As a bonus, we will also see how to visualize linear transformation in Python!

8. [Singular Value Decomposition](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.8-Singular-Value-Decomposition/)

    <img src="2.8 Singular Value Decomposition/images/output_35_7.png" width="300" alt="output_35_7">

    We will see another way to decompose matrices: the Singular Value Decomposition or SVD. Since the beginning of this series I emphasized the fact that you can see matrices as linear transformation in space. With the SVD, you decompose a matrix in three other matrices. We will see that we look at these new matrices as *sub-transformation* of the space. Instead of doing the transformation in one movement, we decompose it in three movements. As a bonus, we will apply the SVD to image processing. We will see the effect of SVD on an example image of Lucy the goose. So keep on reading!

9. [The Moore-Penrose Pseudoinverse](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.9-The-Moore-Penrose-Pseudoinverse/)

    <img src="2.9 The Moore-Penrose Pseudoinverse/images/overdetermined-system-equations-python.png" width="300" alt="output_44_0">

    We saw that not all matrices have an inverse. It is unfortunate because the inverse is used to solve system of equations. In some cases, a system of equations has no solution, and thus the inverse doesn’t exist. However it can be useful to find a value that is almost a solution (in terms of minimizing the error). This can be done with the pseudoinverse! We will see for instance how we can find the best-fit line of a set of data points with the pseudoinverse.

10. [The Trace Operator](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.10-The-Trace-Operator/)

    <img src="2.10 The Trace Operator/images/trace-matrix.png" width="200" alt="Calculating the trace of a matrix" title="Calculating the trace of a matrix">

    **The trace of matrix**

    We will see what is the Trace of a matrix. It will be needed for the last chapter on the Principal Component Analysis (PCA).

11. [The Determinant](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.11-The-determinant/)

    <img src="2.11 The determinant/images/positive-negative-determinant.png" width="400" alt="Comparison of positive and negative determinant" title="Comparison of the effect of positive and negative determinants">

    **Link between the determinant of a matrix and the transformation associated with it**

    This chapter is about the determinant of a matrix. This special number can tell us a lot of things about our matrix!

12. [Example: Principal Components Analysis](https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.12-Example-Principal-Components-Analysis/)

    <img src="2.12 Example - Principal Components Analysis/images/gradient-descent.png" width="400" alt="Mechanism of the gradient descent algorithm" title="Mechanism of the gradient descent algorithm">
    **Gradient descent**

    This is the last chapter of this series on linear algebra! It is about Principal Components Analysis (PCA). We will use some knowledge that we acquired along the preceding chapters to understand this important data analysis tool!


# Feedback

If you find errors/misunderstandings/typos… Please report it! You can send me emails or open issues and pull request in the notebooks Github.

# References

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.

[Source](https://github.com/hadrienj/deepLearningBook-Notes)
