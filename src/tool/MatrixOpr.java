package tool;

import java.util.*;

public class MatrixOpr {
	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	 */
	public static float[][] Matrix_Mult(float[][] paraU, float[][] paraV) {
		float[][] tempResultMatrix = new float[paraU.length][paraV.length];
		if (paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraV.length; j++) {
					for (int k = 0; k < paraU[0].length; k++) {
						tempResultMatrix[i][j] += paraU[i][k] * paraV[j][k];
					} // of for k
				} // of for j
			} // of for i
		}// of if

		return tempResultMatrix;
	}// of Matrix_Multiply

	public static float[][] ColVec_Multi_RowVec(float[] paraColVec,
			float[] paraRowVec) {
		float[][] tempResultMatrix = new float[paraColVec.length][paraRowVec.length];
		for (int i = 0; i < paraColVec.length; i++) {
			for (int j = 0; j < paraRowVec.length; j++) {
				tempResultMatrix[i][j] = paraColVec[i] * paraRowVec[j];
			} // of for j
		} // of for i
		return tempResultMatrix;
	}// of Matrix_Multiply

	public static float Vector_Mult(float[] paraU, float[] paraV) {
		float tempResult = 0;
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResult += paraU[i] * paraV[i];
			} // of for i
		} // of if

		return tempResult;
	}// of dotMultiply

	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	 */
	public static float[][] Matrix_DotMult(float[][] paraU, float[][] paraV) {
		float[][] tempResultMatrix = new float[paraU.length][paraU[0].length];

		if (paraU.length == paraV.length && paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraU[0].length; j++) {
					tempResultMatrix[i][j] = paraU[i][j] * paraV[i][j];
					if (Math.abs(tempResultMatrix[i][j]) == 0.0 // To check
																// whether
																// tempResultMatrix[i][j]
																// is equal to
																// -0.0
							&& Math.copySign(1.0, tempResultMatrix[i][j]) < 0.0) {
						tempResultMatrix[i][j] = 0;
					}// of if
				} // of for j
			} // of for i
		}// of if
		return tempResultMatrix;
	}// of Matrix_DotMult

	public static float[] Vector_DotMult(float[] paraU, float[] paraV) {
		float[] tempResultVector = new float[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResultVector[i] = paraU[i] * paraV[i];
			} // of for i
		} // of if
		return tempResultVector;
	}// of Vector_DotMult

	public static float[][] Matrix_DotDiv(float[][] paraU, float[][] paraV) {
		float[][] tempResultVector = new float[paraU.length][paraU[0].length];
		if (paraU.length == paraV.length && paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraU[0].length; j++) {
					if (paraV[i][j] != 0) {
						tempResultVector[i][j] = paraU[i][j] / paraV[i][j];
					}// of if
				}// of for j
			} // of for i
		} // of if
		return tempResultVector;
	}// of Vector_DotDiv

	public static float[] Vector_DotDiv(float[] paraU, float[] paraV) {
		float[] tempResultVector = new float[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				if (paraV[i] != 0) {
					tempResultVector[i] = paraU[i] / paraV[i];
				}// of if
			} // of for i
		} // of if
		return tempResultVector;
	}// of Vector_DotDiv

	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	 */
	public static double[][] Matrix_Add(double[][] paraU, double[][] paraV) {
		double[][] tempResultMatrix = new double[paraU.length][paraV.length];

		for (int i = 0; i < paraU.length; i++) {
			for (int j = 0; j < paraU[0].length; j++) {
				tempResultMatrix[i][j] = paraU[i][j] + paraV[i][j];
			} // of for j
		} // of for i

		return tempResultMatrix;
	}// of add

	public static float[][] Matrix_Sub(float[][] paraU, float[][] paraV) {
		float[][] tempResultMatrix = new float[paraU.length][paraV[0].length];
		if (paraU.length == paraV.length && paraU[0].length == paraV[0].length) {
			for (int i = 0; i < paraU.length; i++) {
				for (int j = 0; j < paraU[0].length; j++) {
					tempResultMatrix[i][j] = paraU[i][j] - paraV[i][j];
				} // of for j
			} // of for i
		}// of if
		return tempResultMatrix;
	}// of Matrix_Sub

	public static float[] Vector_Sub(float[] paraU, float[] paraV) {
		float[] tempResultVector = new float[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResultVector[i] = paraU[i] - paraV[i];
			} // of for i
		} // of if

		return tempResultVector;
	}// of Vector_Sub

	public static float[] Vector_Add(float[] paraU, float[] paraV) {
		float[] tempResultVector = new float[paraU.length];
		if (paraU.length == paraV.length) {
			for (int i = 0; i < paraU.length; i++) {
				tempResultVector[i] = paraU[i] + paraV[i];
			} // of for i
		} // of if
		return tempResultVector;
	}// of Vector_Sub

	/*
	 * paraVector is considered as a column vector, and is expanded to a matrix.
	 * Each column of the matrix is paraVector. And the column number of the
	 * matrix is paraColNum.
	 */
	public static float[][] Vector2Matrix(float[] paraVector, int paraColNum) {
		float[][] tempResultMatrix = new float[paraVector.length][paraColNum];
		for (int j = 0; j < paraColNum; j++) {
			for (int i = 0; i < paraVector.length; i++) {
				tempResultMatrix[i][j] = paraVector[i];
			}// of for i
		}// of for j
		return tempResultMatrix;
	}// of Vector2Matrix

	public static float[] sumByColumn(float[][] paraMatrix) {
		float[] tempResult = new float[paraMatrix[0].length];
		for (int j = 0; j < paraMatrix[0].length; j++) {
			for (int i = 0; i < paraMatrix.length; i++) {
				tempResult[j] += paraMatrix[i][j];
			}// of for i
		}// of for j
		return tempResult;
	}// of SumCol

	public static float Matrix_Sum(float[][] paraMatrix) {
		float tempResult = 0;

		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[0].length; j++) {
				tempResult += paraMatrix[i][j];
			} // of for j
		} // of for i

		return tempResult;
	}// of Matrix_Sum

	public static float[][] Matrix_Transpose(float[][] paraMatrix) {
		float[][] tempResult = new float[paraMatrix[0].length][paraMatrix.length];

		for (int i = 0; i < tempResult.length; i++) {
			for (int j = 0; j < tempResult[0].length; j++) {
				tempResult[i][j] = paraMatrix[j][i];
			} // of for j
		} // of for i

		return tempResult;
	}// of Matrix_Transpose

	public static float[][] Add_MatrixandNumber(float[][] paraMatrix,
			float paraNum) {
		float[][] tempResult = new float[paraMatrix.length][paraMatrix[0].length];

		for (int i = 0; i < tempResult.length; i++) {
			for (int j = 0; j < tempResult[0].length; j++) {
				tempResult[i][j] = paraMatrix[i][j] + paraNum;
			} // of for j
		} // of for i

		return tempResult;
	}// of Add_MatrixandNumber

	public static float[][] Matrix_Subspace(float[][] paraMatrix1,
			float[][] paraMatrix2) {
		float[][] tempResult;
		float[][] tempMulti;
		// double tempMax = 0;
		// if (paraMatrix1.length == paraMatrix2.length && paraMatrix1[0].length
		// == paraMatrix2[0].length) {

		tempMulti = new float[paraMatrix1[0].length][paraMatrix1[0].length];
		tempResult = new float[tempMulti.length][tempMulti[0].length];

		tempMulti = Matrix_Mult(paraMatrix1, paraMatrix2);

		for (int i = 0; i < tempMulti.length; i++) {
			for (int j = 0; j < tempMulti[0].length; j++) {
				tempMulti[i][j] = (float)Math.acos(Math.abs(tempMulti[i][j]));
			}// of for j
		}// of for i

		tempResult = tempMulti;
		return tempResult;
		// }//of if

		// return tempMulti;
	}// of Matrix_Subspace

	public static double getMedian(float[] arr) {
		float[] tempArr = Arrays.copyOf(arr, arr.length);
		Arrays.sort(tempArr);
		if (tempArr.length % 2 == 0) {
			return (tempArr[tempArr.length >> 1] + tempArr[(tempArr.length >> 1) - 1]) / 2;
		} else {
			return tempArr[(tempArr.length >> 1)];
		}
	}// of getMedian

	/**
	 * 
	 * @param paraU
	 * @param paraV
	 * @return
	public static double[][] Matrix_Col_Mult(double[][] paraU, double[][] paraV, int paraFirstCol, int paraSecond) {
		for (int i = 0; i < paraU.length; i++) {
			paraU[i][paraFirstCol]
		}
		return null;
	}
	}
	 */
	/**
	 ***********************
	 * Print the matrix
	 ***********************
	 */
	public static void printMatrix(double[][] paraMatrix) {
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				System.out.printf("|%8.5f|", paraMatrix[i][j]);
			} // Of for j
			System.out.print("*");
			System.out.println();
		} // Of for i
	}// of printMatrix
	
}// of MatrixOpr
