package SingleGaussMagicBarrier;

import MoGMagicBarrier.Iteration;
import MoGMagicBarrier.NormalDistribution;
import datamodel.DataInfo;
import tool.SimpleTool;

public class CondProbSingleSigma {
	public DataInfo data;

	/**
	 * Sigma
	 */
	public double sigma;

	/**
	 * Conditional probability based “true” rating For example: 5 * 5 (LEVEL *
	 * LEVEL)
	 */
	double[][] trueBasedCondPro;

	/**
	 * To count rating distribution, we need to obtain the rating number of each
	 * scale.
	 */

	public double totalGammaOfVisRating; //
	public double[] gammaOfEachVisRating; // LEVEL
	public double[] propOfEachVisRating; // LEVEL
	public double[] propOfTrueRating; // LEVEL

	/**
	 * Conditional probability based "visible" rating
	 */
	public double[][] visBasedCondPro; // LEVEL * LEVEL

	/**
	 * Constructor
	 */
	public CondProbSingleSigma(DataInfo paraData) {
		data = paraData;
		trueBasedCondPro = new double[data.LEVEL][data.LEVEL];
		visBasedCondPro = new double[data.LEVEL][data.LEVEL];
		totalGammaOfVisRating = 0;
		gammaOfEachVisRating = new double[data.LEVEL];
		propOfEachVisRating = new double[data.LEVEL];
		propOfTrueRating = new double[data.LEVEL];
	}// of the first constructor

	/**
	 * Get index based on the rating
	 */
	public int getIndexBasedRating(double paraRating) {
		int tempIndex = 0;

		for (int i = 0; i < data.LEVEL; i++) {
			double tempLow = data.rlow + i * data.stepLen - data.stepLen / 2;
			double tempHigh = data.rlow + i * data.stepLen + data.stepLen / 2;

			if (paraRating > tempLow && paraRating <= tempHigh) {
				tempIndex = i;
				break;
			} // of if
		} // of for i

		return tempIndex;
	}// of getIndexBasedRating

	/**
	 * Compute true-rating-based condition probability with step-length integration
	 * 
	 * output: trueBasedCondPro (LEVEL * LEVEL)
	 */
	public double[][] computeTrueBasedCondProWithInteg(double paraSigma) {
		sigma = Math.sqrt(paraSigma);

		for (int z = 0; z < data.LEVEL; z++) {
			// stepLengthIntegration(i + 1, paraSigma, -1000, 0)
			for (int y = 0; y < data.LEVEL; y++) {
				/*
				 * if (y == 0) { trueBasedCondPro[k][y][z] =
				 * NormalDistribution.stepLengthIntegration(rlow + z * stepLen, sigma[k], -1000,
				 * rlow + stepLen / 2);//1.5 } else if (y == LEVEL - 1) {
				 * trueBasedCondPro[k][y][z] = NormalDistribution.stepLengthIntegration(rlow + z
				 * * stepLen, sigma[k], rhigh - stepLen / 2, 1006); } else {
				 * trueBasedCondPro[k][y][z] = NormalDistribution.stepLengthIntegration(rlow + z
				 * * stepLen, sigma[k], rlow + y * stepLen - stepLen / 2, rlow + y * stepLen +
				 * stepLen / 2); //? } // Of if
				 */
				trueBasedCondPro[y][z] = NormalDistribution.stepLengthIntegration(data.rlow + z * data.stepLen, sigma,
						data.rlow + y * data.stepLen - data.stepLen / 2,
						data.rlow + y * data.stepLen + data.stepLen / 2); // ?
			} // Of for j
		} // Of for i

		// ???归一化
		double tempTotalProbability = 0;

		for (int z = 0; z < data.LEVEL; z++) {
			tempTotalProbability = 0;
			for (int y = 0; y < data.LEVEL; y++) {
				tempTotalProbability += trueBasedCondPro[y][z];
			} // Of for j
				// System.out.println(i + "-tempTotalProbability: " +
				// tempTotalProbability);
			for (int y = 0; y < data.LEVEL; y++) {
				trueBasedCondPro[y][z] /= tempTotalProbability;
				trueBasedCondPro[y][z] = Math.round(trueBasedCondPro[y][z] * 10000) / 10000.0;
			} // Of for j
		} // Of for i

		return trueBasedCondPro;
	}// Of computeTrueBasedCondProWithInteg

	/**
	 * For single sigma, noiseDistribution = 1 for each rating Compute probability
	 * of visual rating
	 * 
	 * output: propOfEachVisRating (numNoise * LEVEL)
	 */
	public void computePropOfVisRating() {
		// System.out.println(paraData.trainVector + "test 0");
		// SimpleTool.printDoubleArray(paraData.trainVector);
		// System.out.println(paraData.trainVector + "test 1");
		// SimpleTool.printFloatArray(paraData.trainVector);
		// SimpleTool.printMatrix(paraMLG.noiseDistribution);
		// System.out.println("rlow: " + rlow + " stepLen: " + stepLen);
		for (int i = 0; i < data.numOfRatings; i++) {
			double tempOrigRating = data.trData[i].rate;

			totalGammaOfVisRating += 1; // paraMLG.noiseDistribution[i][k];
			for (int y = 0; y < data.LEVEL; y++) {
				if (tempOrigRating > data.rlow + y * data.stepLen - data.stepLen / 2.0
						&& tempOrigRating <= data.rlow + y * data.stepLen + data.stepLen / 2.0) {
					gammaOfEachVisRating[y] += 1; // paraMLG.noiseDistribution[i][k];
				} // Of if
			} // of for j
		} // of for i

		for (int y = 0; y < data.LEVEL; y++) {
			propOfEachVisRating[y] = gammaOfEachVisRating[y] / totalGammaOfVisRating;
		} // of for y

		// System.out.println("The proportions of visible rating: ");
		// printDoubleArray(propOfVisRating);
	}// Of computePropOfVisRating

	/**
	 * transposition
	 */
	public double[][] transposition(double[][] paraMatrix) {
		double[][] tempMatrix = new double[paraMatrix[0].length][paraMatrix.length];
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[0].length; j++) {
				tempMatrix[j][i] = paraMatrix[i][j];
			} // Of for j
		} // Of for i
		return tempMatrix;
	}// Of transposition

	/**
	 * Gauss-seidel iteration
	 * 
	 * output: propOfTrueRating[LEVEL]
	 */
	public void computePropOfTrueRatingSOR() {

		for (int z = 0; z < data.LEVEL; z++) {
			propOfTrueRating[z] = 1.0 / data.LEVEL;
		} // Of if

		// double[][][] trueBasedCondPro2 = new double[numNoise][][];

		// trueBasedCondPro2[k] = transposition(trueBasedCondPro[k]);
		// input: trueBasedCondPro[k], LEVEL * LEVEL
		// input: propOfEachVisRating[k], LEVEL
		// output: propOfTrueRating[k], LEVEL
		// propOfTrueRating[k] = Iteration.SOR(trueBasedCondPro[k],
		// propOfEachVisRating[k], 0.1, 1e-3);
		// propOfTrueRating[k] =
		// GuassSeidelInteration.Gauss_seidel2(trueBasedCondPro[k],
		// propOfTrueRating[k], propOfEachVisRating[k], 1e-5);
		propOfTrueRating = Iteration.SOR(trueBasedCondPro, propOfEachVisRating, 0.1, 1e-5);

		System.out.println("computePropOfTrueRatingSOR propOfTrueRating");
		SimpleTool.printDoubleArray(propOfTrueRating);
		// Step 4. Check
		double[] tempPropOfVisibleRating = new double[data.LEVEL];
		for (int z = 0; z < data.LEVEL; z++) {
			for (int y = 0; y < data.LEVEL; y++) {
				tempPropOfVisibleRating[y] += propOfTrueRating[z] * trueBasedCondPro[y][z];
			} // Of for j
		} // Of for i

		// Step 5. If propOfTrueRating < 0, then propOfTrueRating = 1e-4
		for (int z = 0; z < data.LEVEL; z++) {
			if (propOfTrueRating[z] < 1e-6) {
				propOfTrueRating[z] = 1e-4;
			} // Of if
		} // Of for i

		// Step 6. normalization
		double tempTotalProp = 0;
		for (int z = 0; z < data.LEVEL; z++) {
			tempTotalProp += propOfTrueRating[z];
		} // of for i
		for (int z = 0; z < data.LEVEL; z++) {
			propOfTrueRating[z] /= tempTotalProp;
			propOfTrueRating[z] = Math.round(propOfTrueRating[z] * 10000) / 10000.0;
		} // of for i

		// System.out
		// .println("The proportions of true rating (afer normalization):
		// ");
		// printDoubleArray(propOfTrueRating);

		// Step 7. Check 2
		for (int y = 0; y < data.LEVEL; y++) {
			tempPropOfVisibleRating[y] = 0;
			for (int z = 0; z < data.LEVEL; z++) {
				tempPropOfVisibleRating[y] += propOfTrueRating[z] * trueBasedCondPro[y][z];
			} // Of for j
		} // Of for i
		System.out.println("---------------");
		SimpleTool.printDoubleArray(propOfEachVisRating);
		SimpleTool.printDoubleArray(tempPropOfVisibleRating);
		System.out.println("---------------");
	}// Of computePropOfTrueRatingGaussSeidel

	/**
	 * Employ Bayes method to compute visible-rating based conditional probability
	 * 
	 */
	public void computeVisBasedCondProBasedOnBayes() {
		// Step 1. Employ Bayes method to compute conditional probability based
		// on real rating
		for (int y = 0; y < visBasedCondPro.length; y++) {
			for (int z = 0; z < visBasedCondPro[0].length; z++) {
				visBasedCondPro[z][y] = propOfTrueRating[z] * trueBasedCondPro[y][z] / propOfEachVisRating[y];// switching
																												// problem?:
																												// i,j
			} // of for j
		} // Of for i
			// SimpleTool.printDoubleArray(propOfTrueRating[k]);
			// SimpleTool.printMatrix(visBasedCondPro[k]);

		// 归一化:P_{σ_k}(o_{i,j}= z|r_{i,j} = y), 在r_{i,j} = y不变的情况下，o_{i,j}= z的总概率为1
		double tempProbability = 0;
		for (int y = 0; y < data.LEVEL; y++) {//y: visual rating
			tempProbability = 0;
			for (int z = 0; z < data.LEVEL; z++) {//z: true rating
				tempProbability += visBasedCondPro[z][y];
			} // Of for j
			for (int z = 0; z < data.LEVEL; z++) {
				visBasedCondPro[z][y] /= tempProbability;
				visBasedCondPro[z][y] = Math.round(visBasedCondPro[z][y] * 10000) / 10000.0;
			} // Of for j
		} // Of for i
	}// Of computeRealBasedCondProbabilityBasedOnBayes
	
	/**
	 * 
	 * @param args
	 */
	public static void main(String args[]) {
		try {
			// Prepare data and preprocessing
			String tempPropertyFileName = new String("src/properties/ml-100k.properties");
			DataInfo tempData = new DataInfo(tempPropertyFileName);
			tempData.readData();
			tempData.computeDataVector();
			CondProbSingleSigma tempCon = new CondProbSingleSigma(tempData);
			float tempSigma = (float)(0.9 * 0.9);
			//step 1. Calculate the conditional probability P_{σ_k}(r_{i,j}= y|o_{i,j} = z)
			tempCon.computeTrueBasedCondProWithInteg(tempSigma);
			System.out.println("P_{σ_k}(r_{i,j}= y|o_{i,j} = z) is: ");
			SimpleTool.printMatrix(tempCon.trueBasedCondPro);
			//step 2. Calculate the probability P_{σ_k}(r_{i,j} = y)
			tempCon.computePropOfVisRating();
			System.out.println("P_{σ_k}(r_{i,j} = y is: ");
			SimpleTool.printDoubleArray(tempCon.propOfEachVisRating);
			//step 3. Calculate the probability P_{σ_k}(o_{i,j} = z)
			tempCon.computePropOfTrueRatingSOR();
			System.out.println("P_{σ_k}(o_{i,j} = z) is: ");
			SimpleTool.printDoubleArray(tempCon.propOfTrueRating);
			//step 4. calculate the probability P_{σ_k}(o_{i,j}= z|r_{i,j} = y)
			tempCon.computeVisBasedCondProBasedOnBayes();
			System.out.println("P_{σ_k}(o_{i,j}= z|r_{i,j} = y) is: ");
			SimpleTool.printMatrix(tempCon.visBasedCondPro);	
		} catch (Exception e) {
			e.printStackTrace();
		} // of try
	}//of main
}// of class ConditionalProbability
