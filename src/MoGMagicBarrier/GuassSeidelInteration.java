package MoGMagicBarrier;

import java.util.Arrays;

public class GuassSeidelInteration {
	/**
	 * Compare two arrays
	 * 
	 * @param paraA
	 * @param paraB
	 * @return
	 */
	public static double Compare(double[] paraA, double[] paraB) {
		double tempValue = 0;
		int i;
		for (i = 0; i < paraA.length; i++) {
			tempValue += Math.abs(paraA[i] - paraB[i]);
		} // Of for i
		return tempValue;
	}// Of Compare

	/**
	 * paraA * paraX = paraB trueBasedCondPro * propOfTrueRating =
	 * propOfEachVisRating
	 * 
	 * @param paraA:     trueBasedCondPro(y,z), LEVEL * LEVEL
	 * @param paraX:     propOfTrueRating(z), LEVEL
	 * @param paraB:     propOfEachVisRating(y), LEVEL
	 * @param precesion: 1e-5
	 */
	public static double[] Gauss_seidel(double[][] paraA, double[] paraX, double[] paraB, double precesion) {
		double[] x2 = new double[paraX.length];
		double[] x3 = new double[paraX.length];
		double sum;
		for (int z = 0; z < paraX.length; z++) {
			x2[z] = paraX[z];
			x3[z] = paraX[z];
		} // of for i
		int k = 1; // k 为迭代次数
		while (true) {
			for (int z = 0; z < paraA.length; z++) {
				sum = 0;
				for (int y = 0; y < paraA[0].length; y++) {
					if (y != z) {
						sum += paraA[z][y] * x2[z];
					} // Of if
				} // Of for y
				paraX[z] = (paraB[z] - sum) / paraA[z][z];
				x2[z] = paraX[z];
			} // of for z
			/*
			 * // 输出每一次迭代的结果 System.out.println("第%d 次迭代:\n" + k);
			 * System.out.println("x3= "); for (i = 0; i < paraX.length; i++) {
			 * System.out.print(" " + x3[i]); }// Of for i System.out.println();
			 * System.out.println("x= "); for (i = 0; i < paraX.length; i++) {
			 * System.out.print(" " + paraX[i]); }// Of for i System.out.println();
			 */
			// 判断是否达到迭代精度
			if (Compare(x3, paraX) < precesion) {
				/*
				 * System.out.println("达到迭代精度的方程组的解为:\n"); System.out.println("x= "); for (i =
				 * 0; i < paraX.length; i++) { System.out.print(" " + paraX[i]); }// Of for i
				 * System.out.println();
				 */
				break;
			} else {
				for (int z = 0; z < paraX.length; z++) {
					x3[z] = paraX[z];
				} // Of for i
				k++;
				continue;
			} // of if
		} // of while

		return paraX;
	}// Of Gauss_seidel

	/**
	 * paraA * paraX = paraB
	 * 
	 * @param paraA
	 * @param paraX
	 * @param paraB
	 * @param precesion
	 */
	public static double[] Gauss_seidel2(double[][] paraA, double[] paraX, double[] paraB, double precesion) {
		int i, j, k;
		double[] x2 = new double[paraX.length];
		double[] x3 = new double[paraX.length];
		double sum;
		for (i = 0; i < paraX.length; i++) {
			x2[i] = paraX[i];
			x3[i] = paraX[i];
		} // of for i
		k = 1; // k 为迭代次数
		while (true) {
			for (i = 0; i < paraX.length; i++) {
				sum = 0;
				for (j = 0; j < paraX.length; j++) {
					if (j != i) {
						sum += paraA[i][j] * x2[j];
					} // Of if
				} // Of for j
				paraX[i] = (paraB[i] - sum) / paraA[i][i];
				x2[i] = paraX[i];
			} // of for i
			/*
			 * // 输出每一次迭代的结果 System.out.println("第%d 次迭代:\n" + k);
			 * System.out.println("x3= "); for (i = 0; i < paraX.length; i++) {
			 * System.out.print(" " + x3[i]); }// Of for i System.out.println();
			 * System.out.println("x= "); for (i = 0; i < paraX.length; i++) {
			 * System.out.print(" " + paraX[i]); }// Of for i System.out.println();
			 */
			// 判断是否达到迭代精度
			if (Compare(x3, paraX) < precesion) {
				/*
				 * System.out.println("达到迭代精度的方程组的解为:\n"); System.out.println("x= "); for (i =
				 * 0; i < paraX.length; i++) { System.out.print(" " + paraX[i]); }// Of for i
				 * System.out.println();
				 */
				break;
			} else {
				for (i = 0; i < paraX.length; i++) {
					x3[i] = paraX[i];
				} // Of for i
				k++;
				continue;
			} // of if
		} // of while

		return paraX;
	}// Of Gauss_seidel

	public static void main(String[] args) {
		// TODO 自动生成的方法存根
		double[][] A = { { 8, -3, 2 }, { 4, 11, -1 }, { 6, 3, 12 } };
		double[] x = { 0, 0, 0 };
		double[] b = { 20, 33, 36 };
		GuassSeidelInteration tempGs = new GuassSeidelInteration();
		double[] finalResult = tempGs.Gauss_seidel(A, x, b, 1e-6);
		System.out.println(Arrays.toString(finalResult));
	}// Of main
}// Of GuassSeidelInteration
