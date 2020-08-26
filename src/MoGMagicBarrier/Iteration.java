package MoGMagicBarrier;

/**
 * 用迭代法求解线性方程组： Jacobi、Gauss-Seidel、SOR
 * 
 * @author KeXin
 *
 */
public class Iteration {
	/**
	 * Jacobi迭代法
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] Jacobi(double[][] a, double[] b) {
		int n = a.length;
		double sum = 0;
		double e = 0.001;
		double z;
		int i;
		double[] x = new double[n];
		double[] y = new double[n];
		while (true) {
			// 按迭代式迭代
			for (i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					if (i != j) {
						sum += a[i][j] * x[j];
					}
				}
				y[i] = (b[i] - sum) / a[i][i];
				sum = 0;
			}
			// 达到精度后终止
			i = 0;
			while (i < n) {
				z = Math.abs(x[i] - y[i]);
				if (z > e)
					break;
				i++;
			}
			if (i != n) {
				for (i = 0; i < n; i++)
					x[i] = y[i];
			} else if (i == n)
				break;
		}
		return y;
	}

	/**
	 * 高斯-赛德尔方法
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] GaussSeidel(double[][] a, double[] b, double e) {
		int n = a.length;
		double sum = 0;
		// double e = 0.001;
		double z;
		int i;
		double[] x = new double[n];
		double[] y = new double[n];
		while (true) {
			// 按迭代式迭代
			for (i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					// 跟Jacobi的不同之处
					if (i < j) {
						sum += a[i][j] * y[j];
					} else if (i > j) {
						sum += a[i][j] * x[j];
					}
				}
				y[i] = (b[i] - sum) / a[i][i];
				sum = 0;
			}
			// 达到精度后终止
			i = 0;
			while (i < n) {
				z = Math.abs(x[i] - y[i]);
				if (z > e)
					break;
				i++;
			}
			if (i != n) {
				for (i = 0; i < n; i++)
					x[i] = y[i];
			} else if (i == n)
				break;
		}
		return y;
	}

	/**
	 * 逐次超松弛方法SOR
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static double[] SOR(double[][] a, double[] b, double m, double e) {
		int n = a.length;
		double sum = 0;
		// double e = 0.001;
		double z;
		int i;
		double[] x = new double[n];
		double[] y = new double[n];
		while (true) {
			// 按迭代式迭代
			for (i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					// 跟Jacobi的不同之处
					if (i < j) {
						sum += a[i][j] * y[j];
					} else if (i > j) {
						sum += a[i][j] * x[j];
					}
				}
				y[i] = (1 - m) * x[i] + (m * (b[i] - sum)) / a[i][i];
				//Modified by henry (because y[i] is the output probability, y[i] >=0,2020-08-05)
				if(y[i] < 1e-6) {
					y[i] = 1e-4;
				}//of if
				sum = 0;
			}
			// 达到精度后终止
			i = 0;
			while (i < n) {
				z = Math.abs(x[i] - y[i]);
				if (z > e)
					break;
				i++;
			}
			if (i != n) {
				for (i = 0; i < n; i++)
					x[i] = y[i];
			} else if (i == n)
				break;
		}
		return y;
	}

	/**
	 * 打印数组
	 * 
	 * @param result
	 */
	public static void PrintArray(String str, double[] result) {
		int n = result.length;
		System.out.print(str + "\n[");
		for (int i = 0; i < n; i++) {
			System.out.print(result[i] + "\t");
		}
		System.out.print(']');
		System.out.println();
	}

	public static void main(String[] args) {
		// Jacobi迭代
		double[][] j1 = { { 4, -1, 1 }, { 4, -8, 1 }, { -2, 1, 5 } };
		double[] b1 = { 7, -21, 15 };
		double[] result1 = Jacobi(j1, b1);
		PrintArray("Jacobi迭代：", result1);
		// Gauss-Seidel迭代
		double[] result = GaussSeidel(j1, b1, 1e-5);
		PrintArray("Gauss-Seidel迭代：", result);
		// SOR迭代
		double m = 0.5;
		double[] result2 = SOR(j1, b1, m, 1e-3);
		PrintArray("m=" + m + " SOR迭代：", result2);
		// 交换第一个和最后一个方程
		double[][] j2 = { { -2, 1, 5 }, { 4, -8, 1 }, { 4, -1, 1 } };
		double[] b2 = { 15, -21, 7 };
		double[] result3 = Jacobi(j2, b2);
		PrintArray("交换方程之后Jacobi迭代：", result3);
	}

}
