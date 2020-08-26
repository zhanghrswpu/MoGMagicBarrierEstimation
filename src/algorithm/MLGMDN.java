package algorithm;

import java.io.IOException;

import datamodel.DataInfo;

//import Jama.*;

//import datamodel.NoiseInfo;
import tool.MatrixOpr;
import tool.SimpleTool;

/**
 * <p>
 * Summary: Deyu Meng, Fernando De la Torre. Robust matrix factorization with unknown noise, ICCV, 2013.
 * Perform EM algorithm for fitting the MoG model.
 * Step 1: Max parameters;
 * Step 2: Expectation;
 * Step 3: Max weighted L2 MF;
 * Step 4: Expectation.
 * <p>
 * Author: <b>Henry</b> zhanghrswpu@163.com <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * SouthWest Petroleum University, Sichuan 610500, China.<br>
 * Progress: OK. Copied from Hydrosimu.<br>
 * Written time: August 15, 2017. <br>
 * Last modify time: August 15, 2017.
 */

public class MLGMDN {
	DataInfo data;

	/**
	 * The mean values of Gaussian distributions.
	 */
	public float[] mu;

	/**
	 * The variances of Gaussian distributions.
	 */
	public float[] sigma;

	/**
	 * The number of users.
	 */
	int numUsers;

	/**
	 * The number of items.
	 */
	int numItems;

	/**
	 * The number of non-zero elememts in the training set.
	 */
	int numTrainSize;

	/**
	 * The rank of factorization matrices.
	 */
	public int rank;

	/**
	 * The type of noises.
	 */
	public int numNoise;

	/**
	 * The weight of every noise. It is \Pi in the paper. Elements are \pi_k
	 */
	public float[] noiseWeight;

	/**
	 * Record the error, the length is the size of training set.
	 */
	float[] trainingErrors;

	/**
	 * Help computing noiseDistribution. The size is the same as
	 * noiseDistribution.
	 */
	float[][] logRho;
	
	/**
	 * A latent variable in the model, with z_{ijk} \in {0, 1}
	 */
	public int[][] z;

	/**
	 * @see logRho
	 */

	float[] logSumExp;

	/**
	 * The log of likelihood. It is the sum of logSumExp divided by the
	 * numTrainSize. Compute in expectation().
	 */
	float logLikelihood;

	/**
	 * The iteration times of matrix factorization
	 */
	public int decompIterTimes;

	/**
	 * The convergence condition
	 */
	public float tolerance;

	/**
	 * A parameter in efficientMCL2
	 */
	float[][] Effic_para1;

	/**
	 * The square roots of noiseWeight matrix
	 */
	float[][] sqrtW;

	/**
	 * The two matrices of factorization results
	 */
	public float[][] factor_U;

	/**
	 * The two matrices of factorization results
	 */
	public float[][] factor_V;

	/**
	 * The noise distribution along the training set. The number of columns is
	 * the number of noise. It is \gamma_{ijk}, where ij is given by the row,
	 * and k is given by the column.
	 */
	public float[][] noiseDistribution;

	/**
	 * Record the likelihood of each iteration
	 */
	public float[] llh;

	/**
	 * The predicted rating matrix.
	 */
	public float[][] predictions;

	/**
	 **********************
	 * @param paraData
	 ********************** 
	 */
	public MLGMDN(DataInfo paraData) {
		rank = paraData.rank;
		numNoise = paraData.numNoise;
		
		data = paraData;
		initialize();
	}// Of the constructor

	/**
	 ***********************
	 * Initialize parameters
	 ***********************
	 */
	public void initialize() {
		numUsers = data.userNum;
		numItems = data.itemNum;
		
		//factor_U = new float[numUsers][rank];
		//factor_V = new float[numItems][rank];

		numTrainSize = data.numOfRatings;
		
		//data.generateRandomSubMatrix(rank);
		factor_U = data.subU;
		factor_V = data.subV;
		
		// Share the same space of data.
		//data.setRandomNoiseDistribution(numNoise);
		noiseDistribution = data.getNoiseDistribution();
		mu = new float[numNoise];
		sigma = new float[numNoise];
		noiseWeight = new float[numNoise];

		trainingErrors = new float[numTrainSize];
		logLikelihood = 0;
		decompIterTimes = data.MFIterTimes;
		tolerance = (float)1e-8;
		Effic_para1 = new float[numUsers][numItems];
		sqrtW = new float[numUsers][numItems];

		System.out.print("Initialize sigma = ");

		for (int i = 0; i < sigma.length; i++) {
			// sigma[i] = 0.4 * (i + 1);//Math.random();
			sigma[i] = (float)Math.random();
			// System.out.println("sigma" + i + ": " + sigma[i]);
			mu[i] = 0;
			noiseWeight[i] = data.noiseWeight[i];
		} // of for i
		SimpleTool.printFloatArray5(sigma);

		logRho = new float[numTrainSize][numNoise];// 640*3
		z = new int[numTrainSize][numNoise];
	}// of initialize

	/**
	 ***********************
	 * Compute the error on non-zero points of the training set. Note that the
	 * error can be either positive or negative.
	 * 
	 * @return the train errors in an array with size of the training set.
	 ***********************
	 */
	public float[] computeNonZeroError() {
		float[][] tempX = MatrixOpr.Matrix_Mult(factor_U, factor_V);
		int tempCount = 0;

		for (int i = 0; i < data.uRatings.length; i++) {
			for (int j = 0; j < data.uRatings[i].length; j++) {
				trainingErrors[tempCount] = data.uRatings[i][j]
						- tempX[i][data.uRateInds[i][j]];
				tempCount++;
			} // of for j
		} // of for i

		// System.out.println("The length of trainingErrors: " +
		// trainingErrors.length);
		return trainingErrors;
	}// of computeNonZeroError

	/**
	 ***********************
	 * The E step of the EM algorithm. Compute the expected noise for each data
	 * point. Recompute logRho, noiseDistribution and logLikelihood. Update the
	 * noiseDistribution
	 * 
	 * @return noiseDistribution
	 ***********************
	 */
	public float[][] expectation() {
		// double[][] R = new double[numTrainSize][numNoise];
		float[] tempVector = null;// = new double[numTrainSize];

		// logRho = new double[numTrainSize][numNoise];// 640*3

		// Step 1. logRho(:,i) = logGaussPdf(X,mu(i),Sigma(i));
		for (int j = 0; j < logRho[0].length; j++) {
			tempVector = SimpleTool
					.logGaussPdf(trainingErrors, mu[j], sigma[j]);
			for (int i = 0; i < logRho.length; i++) {
				logRho[i][j] = tempVector[i];
			} // for i
		} // for j

		// System.out.println("logRho");
		// SimpleTool.printMatrix(logRho);

		// Step 2. logRho = bsxfun(@plus,logRho,log(w));
		for (int i = 0; i < logRho.length; i++) {
			for (int j = 0; j < logRho[0].length; j++) {
				logRho[i][j] += Math.log(noiseWeight[j]);
			} // of for j
		} // of for i

		// System.out.println("logRho+logw:");
		// SimpleTool.printMatrix(logRho);

		// Step 3. T = logSumExp(logRho, 2);
		computeLogSumExp();

		// System.out.println("Print logRho again:");
		// SimpleTool.printMatrix(logRho);

		logLikelihood = 0;
		// Step 4. llh = sum(T)/n; % loglikelihood
		for (int i = 0; i < logSumExp.length; i++) {
			logLikelihood += logSumExp[i];
		} // of for i
		logLikelihood = logLikelihood / numTrainSize;

		// Step 5.logR = bsxfun(@minus,logRho,T);
		// Step 6. R = exp(logR);
		for (int i = 0; i < logRho.length; i++) {
			for (int j = 0; j < logRho[0].length; j++) {
				noiseDistribution[i][j] = (float)Math.exp(logRho[i][j] - logSumExp[i]);
			} // of for j
		} // of for i

		return noiseDistribution;
	}// of expectation

	/**
	 ******************* 
	 * Iterate Expectation step and Maximization step. The core code of the
	 * algorithm
	 ******************* 
	 */
	public void iterationEM(int paraMaxIterTimes) {
		boolean converged = false;
		int tempIterations = 0;

		// subU and subV are already initialized
		computeNonZeroError();//计算误差矩阵，只保存有值的地方，用一个向量来保存

		// SimpleTool.printDoubleArray(tempEm.trainingErrors);
		expectation();

		// Initialization
		llh = new float[paraMaxIterTimes];
		llh[0] = logLikelihood;// before iterations
		// System.out.printf("llh[0] = %8.2f\r\n", llh[0]);

		// If converged, do not repeat more
		while (converged != true && tempIterations < paraMaxIterTimes - 1) {
			// System.out.println("Iterating");
			tempIterations++;
			// ******* M Step 1************
			updateNoiseParameters();

			// ******* E Step 1 compute the noise distribution************
			expectation();
			// templlh1 = logLikelihood;
			// llh[tempIterations] = templlh1;

			// ******* M Step 2************
			// Compute subU and subV according to the noise distribution
			maximizationW();

			// Prepare for the noise part
			computeNonZeroError();

			// ******* E Step 2************
			expectation();
			// templlh2 = logLikelihood;
			llh[tempIterations] = logLikelihood;

			if (Math.abs(llh[tempIterations] - llh[tempIterations - 1]) < tolerance
					* Math.abs(llh[tempIterations])) {
				converged = true;
			}// Of if
		} // of while
		
		//Compute the latent variable z_{ijk}
		//modified by henry (20180816)
		double tempMax = -1;
		int tempIndex = -1;
		for(int i = 0; i < noiseDistribution.length; i ++){
			tempMax = -1;
			tempIndex = -1;
			for(int j = 0; j < noiseDistribution[0].length; j ++){
				if(noiseDistribution[i][j] > tempMax){
					tempMax = noiseDistribution[i][j];
					tempIndex = j;
				}//of if
			}//of for j
			
			z[i][tempIndex] = 1;
		}//of for i
		System.out.println("Acutual iteration time: " + tempIterations);
	}// of iteration

	/**
	 ***********************
	 * Compute logSumExp $\ln \sum_{k=1}^K \pi_k N(\mu, \sigma^2)$ using logRho.<br>
	 * Suppose that k = 3 and \log a_2 is the maximal value among {\log a_1,
	 * \log a_2, \log a_3} $\log (a_1 + a_2 + a3) = \log(e^{\log a_1 - \log a_2}
	 * + e^0$ $ + e^{\log a_3 - \log a_2}) + \log a_2$ The purpose is to avoid
	 * computing $a_1$, $a_2$, and $a_3$ which may overflow (too close to 0).
	 * 
	 * @return logSumExp Each element of the vector is obtained using one row of
	 *         logRho
	 ***********************
	 */
	public float[] computeLogSumExp() {
		// Step 1. y = max(x,[],dim) Compute the max value of every row in
		// Matrix logRho.
		float[][] tempXX = new float[logRho.length][logRho[0].length];
		float[] tempRowMax = new float[logRho.length];
		for (int i = 0; i < logRho.length; i++) {
			tempRowMax[i] = logRho[i][0];
			for (int j = 1; j < logRho[0].length; j++) {
				if (tempRowMax[i] < logRho[i][j]) {
					tempRowMax[i] = logRho[i][j];
				} // of if
			} // of for j
		} // of for i

		// Step 2. Compute x = bsxfun(@minus,x,y)
		for (int j = 0; j < logRho[0].length; j++) {
			for (int i = 0; i < logRho.length; i++) {
				tempXX[i][j] = logRho[i][j] - tempRowMax[i];
			} // of for j
		} // of for i

		// Step 3. s = y +log(sum(exp(x),dim)),dim=2,compute the sum of every
		// row.
		float[] tempSum = new float[numTrainSize];
		for (int i = 0; i < logRho.length; i++) {
			tempSum[i] = 0;
			for (int j = 0; j < logRho[0].length; j++) {
				tempSum[i] += Math.exp(tempXX[i][j]);
			} // of for j
			tempSum[i] = (float)Math.log(tempSum[i]);
		} // of for i
		logSumExp = tool.MatrixOpr.Vector_Add(tempRowMax, tempSum);
		// System.out.println("logSumExp:");
		// SimpleTool.printDoubleArray(logSumExp);

		return logSumExp;
	}// of computeLogSumExp

	/**
	 ***********************
	 * Update the parameters of Gaussian distribution for the noise.
	 * Also called "maximizationModel"
	 ***********************
	 */
	public void updateNoiseParameters() {
		float[] tempNk;

		float[] tempDeviations = new float[numTrainSize];
		float[] tempWeightedDeviationSum = new float[numNoise];
		float[] tempsqrtR = new float[numTrainSize];
		
		//Step 1. Recompute noiseWeight nk = sum(R,1); w = nk/size(R,1); 
		//Sigma = zeros(1,k); sqrtR = sqrt(R);
		tempNk = tool.MatrixOpr.sumByColumn(noiseDistribution);

		//Step 2. Recompute mu and sigma mu = zeros(1,k);%fix mu to zero
		for (int i = 0; i < noiseWeight.length; i++) {
			mu[i] = 0;
			sigma[i] = 0;
			noiseWeight[i] = tempNk[i] / numTrainSize;
		} // of for i

		// Step 3. Update sigma (mu = 0?? Henry 20180817)
		for (int j = 0; j < numNoise; j++) {
			for (int i = 0; i < numTrainSize; i++) {
				// To fit vector multiplexion
				tempsqrtR[i] = (float)Math.sqrt(noiseDistribution[i][j]);
				tempDeviations[i] = (trainingErrors[i] - mu[j]) * tempsqrtR[i];
			} // of for i
			tempWeightedDeviationSum[j] = tool.MatrixOpr.Vector_Mult(
					tempDeviations, tempDeviations);
			sigma[j] = (float)(tempWeightedDeviationSum[j] / tempNk[j] + 1e-6);
		} // of for j

		System.out.print("sigma = ");
		SimpleTool.printFloatArray5(sigma);
	}// of updateNoiseParameters

	/**
	 ***********************
	 * Compute two subMatrices of factorization. Prepare/initialize for
	 * efficientMCL2()
	 ***********************
	 */
	public void maximizationW() {
		//double[][] tempW = new double[numUsers][numItems];
		float[][] tempC = new float[numUsers][numItems];

		//Step 1. for j = 1:k W(IND) = W(IND) + R(:,j)/(2*Sigma(j)); 
		//C(IND) = C(IND) + R(:,j)*mu(j)/(2*Sigma(j)); end
		for (int k = 0; k < noiseDistribution[0].length; k++) {
			int tempCnt = 0;
			for(int i = 0; i < data.uRatings.length; i ++){
				for(int j = 0; j < data.uRatings[i].length; j ++){
					if(sigma[k] != 0){
						sqrtW[i][data.uRateInds[i][j]] += noiseDistribution[tempCnt][k]
								/ (2 * sigma[k]);
						Effic_para1[i][data.uRateInds[i][j]] += noiseDistribution[tempCnt][k] * mu[k]
								/ (2 * sigma[k]);// always equal to zero
						tempCnt++;
					}//of if
				}//of for j
			}//of for i
		} // of for k

		Effic_para1 = tool.MatrixOpr.Matrix_DotDiv(Effic_para1, sqrtW);// tempC = 0

		// Step 2. Compute Inx - C and sqrt(W)
		for(int i = 0; i < data.uRatings.length; i ++){
			for(int j = 0; j < numItems; j ++){
				Effic_para1[i][j] = - tempC[i][j];
				sqrtW[i][j] = (float)Math.sqrt(sqrtW[i][j]);
			}//of for j 
			for(int j = 0; j < data.uRatings[i].length; j ++){
				Effic_para1[i][data.uRateInds[i][j]] = data.uRatings[i][j] - tempC[i][data.uRateInds[i][j]];
			}//of for j
		}//of for i
		

		efficientMCL2();
	}// of maximizationW

	/**
	 ***********************
	 * A main algorithm. The low rank matrices are updated decompIterTimes
	 * times.
	 ***********************
	 */
	public void efficientMCL2() {
		int[] randperm;// order randomly

		float[][] tempMulti;
		float[][] tempRegul = new float[numUsers][numItems];// Regulation
		float[][] tempRegul_transp;
		float[][] tempsqrtW_transp;

		float[][] tempPart;

		float[] paraColVec;
		float[] paraRowVec;

		float[][] tempNorm;
		float[] tempVector1 = new float[rank];
		float[] tempVector2 = new float[rank];
		float[] tempVector3 = new float[rank];
		float[][] tempDiag_Sqrt = new float[tempVector1.length][tempVector2.length];
		float[][] tempDiag_U = new float[tempVector1.length][tempVector1.length];
		float[][] tempDiag_V = new float[tempVector2.length][tempVector2.length];
		float[][] tempOutU;
		float[][] tempOutV;
		float[][] tempOutU_transp;
		float[][] tempOutV_transp;

		float[][] paraSubU;
		float[][] paraSubV;
		float[][] paraSubU_transp = new float[numUsers][rank];
		float[][] paraSubV_transp = new float[numItems][rank];
		// paraSubU: InU in efficientMCL2, single direction, not return
		// System.out.println("**************data.subU************");
		// printMatrix(data.subU);
		// System.out.println("**************data.subV************");
		// printMatrix(data.subV);

		paraSubU = data.subU;
		paraSubV = data.subV;

		tempOutU = data.subU;
		tempOutV = data.subV;

		for (int i = 0; i < decompIterTimes; i++) {
			randperm = SimpleTool.generateRandomSequence(rank);
			for (int j = 0; j < randperm.length; j++) {// 1-4 disorder
				// Step 1. TX = Matrix - OutU*OutV' + OutU(:,j)*OutV(:,j)'

				// OutU*OutV'
				tempMulti = tool.MatrixOpr.Matrix_Mult(tempOutU, tempOutV);
																			
				// 4*40
				tempOutU_transp = tool.MatrixOpr.Matrix_Transpose(tempOutU);

				// 4*20
				tempOutV_transp = tool.MatrixOpr.Matrix_Transpose(tempOutV);

				int tempColInd = randperm[j];
				paraColVec = tempOutU_transp[tempColInd];
				paraRowVec = tempOutV_transp[tempColInd];

				// 943*1682
				tempPart = tool.MatrixOpr.ColVec_Multi_RowVec(paraColVec,
						paraRowVec);
				for (int tempi = 0; tempi < tempRegul.length; tempi++) {
					for (int tempj = 0; tempj < tempRegul[0].length; tempj++) {
						tempRegul[tempi][tempj] = Effic_para1[tempi][tempj]
								- tempMulti[tempi][tempj]
								+ tempPart[tempi][tempj];
					} // of for tempj
				} // of for tempi

				// Transpose tempU and tempV for the following computation.

				tempRegul_transp = tool.MatrixOpr.Matrix_Transpose(tempRegul);
				tempsqrtW_transp = tool.MatrixOpr.Matrix_Transpose(sqrtW);// 20*40

				/*
				 * Step 2. u = InU(:,j); OutV(:,j) = optimMCL2(TX,W,u) OutU(:,j)
				 * = optimMCL2(TX',W',OutV(:,j))
				 */
				// System.out.printf("Iteration time: %d,tempColInd:%d\r\n",
				// i,tempColInd);

				paraSubU_transp = tool.MatrixOpr.Matrix_Transpose(paraSubU);// 4*40

				paraSubV_transp = tool.MatrixOpr.Matrix_Transpose(paraSubV);// 4*20

				tempOutV_transp[tempColInd] = optimMCL2(tempRegul, sqrtW,
						paraSubU_transp[tempColInd]);
				// System.out.println("tempColInd:"+tempColInd);
				// System.out.println("**************tempOutV_transp************");
				// printMatrix(tempOutV_transp);

				tempOutU_transp[tempColInd] = optimMCL2(tempRegul_transp,
						tempsqrtW_transp, tempOutV_transp[tempColInd]);
				// System.out.println("**************tempOutU_transp************");
				// printMatrix(tempOutU_transp);

				tempOutU = tool.MatrixOpr.Matrix_Transpose(tempOutU_transp);
				tempOutV = tool.MatrixOpr.Matrix_Transpose(tempOutV_transp);
			} // of for j

			paraSubU = tool.MatrixOpr.Matrix_Transpose(paraSubU_transp);
			paraSubV = tool.MatrixOpr.Matrix_Transpose(paraSubV_transp);

			// Step 3. Compute norm

			tempNorm = tool.MatrixOpr.Matrix_Sub(paraSubU, tempOutU);
			tempNorm = tool.MatrixOpr.Matrix_DotMult(tempNorm, tempNorm);
			double tempSum = tool.MatrixOpr.Matrix_Sum(tempNorm);
			// System.out.printf("Norm is %f\r\n",Math.sqrt(tempSum) );
			if (Math.sqrt(tempSum) < tolerance) {
				break;
			} else {
				paraSubU = tempOutU;
			} // of if
		} // of for i

		/*
		 * Step 4. Nu = sqrt(sum(OutU.^2))'; Nv = sqrt(sum(OutV.^2))'; No
		 * =diag(Nu.*Nv); OutU = OutU*diag(1./Nu)*sqrt(No); OutV =
		 * OutV*diag(1./Nv)*sqrt(No);
		 */
		tempVector1 = tool.MatrixOpr.sumByColumn(tool.MatrixOpr.Matrix_DotMult(
				tempOutU, tempOutU));// 1*4
		tempVector2 = tool.MatrixOpr.sumByColumn(tool.MatrixOpr.Matrix_DotMult(
				tempOutV, tempOutV));// 1*4
		for (int i = 0; i < tempVector1.length; i++) {
			tempVector1[i] = (float)Math.sqrt(tempVector1[i]);
			tempVector2[i] = (float)Math.sqrt(tempVector2[i]);
		} // of for i
		tempVector3 = tool.MatrixOpr.Vector_DotMult(tempVector1, tempVector2);

		for (int i = 0; i < tempDiag_Sqrt.length; i++) {
			tempDiag_Sqrt[i][i] = (float)Math.sqrt(tempVector3[i]);// sqrt(No)
			tempDiag_U[i][i] = (float)(1.0 / tempVector1[i]);// diag(1./Nu)
			tempDiag_V[i][i] = (float)(1.0 / tempVector2[i]);// diag(1./Nv)
		} // of for i

		tempOutU = tool.MatrixOpr.Matrix_Mult(tempOutU,
				tool.MatrixOpr.Matrix_Mult(tempDiag_U, tempDiag_Sqrt));
		tempOutV = tool.MatrixOpr.Matrix_Mult(tempOutV,
				tool.MatrixOpr.Matrix_Mult(tempDiag_V, tempDiag_Sqrt));
		factor_U = tempOutU;
		factor_V = tempOutV;
		// System.out.println("*************************Factorization
		// U*************************");
		// SimpleTool.printMatrix(factor_U);
		// System.out.println("*************************Factorization
		// V*************************");
		// SimpleTool.printMatrix(factor_V);
	}// of efficientMCL2;

	/**
	 ***********************
	 * Compute a column vector.
	 * 
	 * @param paraMatrix
	 * @return The column vector indicating
	 * @see #efficientMCL2()
	 ***********************
	 */
	public static float[] optimMCL2(float[][] paraMatrix, float[][] parasqrtW,
			float[] paraVector) {
		float[][] tempX;
		float[][] tempXX;
		float[] resultOptimVec;
		float[][] tempVec2Matrix;
		float[] tempUp;
		float[] tempDown;

		// Step 1. TX = W.*Matrix, size: equal to W or Matrix,
		tempX = tool.MatrixOpr.Matrix_DotMult(parasqrtW, paraMatrix);

		// Step 3. U = u*ones(1,n), size: paraVector.length
		tempVec2Matrix = tool.MatrixOpr.Vector2Matrix(paraVector,
				paraMatrix[0].length);

		// Step 4. U = W.* U; size:equal to W,
		tempXX = tool.MatrixOpr.Matrix_DotMult(parasqrtW, tempVec2Matrix);

		// Step 5. up = sum(TX.*U), size: the column number of TX
		tempUp = tool.MatrixOpr.sumByColumn(tool.MatrixOpr.Matrix_DotMult(tempX,
				tempXX));

		// Step 6. down = sum (U.* U), size: the column number of U (or TX)
		tempDown = tool.MatrixOpr.sumByColumn(tool.MatrixOpr.Matrix_DotMult(
				tempXX, tempXX));

		// Step 7. v = up ./ down
		resultOptimVec = tool.MatrixOpr.Vector_DotDiv(tempUp, tempDown);
		return resultOptimVec;
	}// of optimMCL2

	/**
	 ***********************
	 * Compute which distribution the noise on every rating belongs to.
	 ***********************
	 */
	public void printNoiseDistribution() {
		int[] belongVector = new int[noiseDistribution.length];
		float[] tempRowMax = new float[noiseDistribution.length];
		for (int i = 0; i < noiseDistribution.length; i++) {
			tempRowMax[i] = noiseDistribution[i][0];
			belongVector[i] = 0;
			for (int j = 1; j < noiseDistribution[0].length; j++) {
				if (tempRowMax[i] < noiseDistribution[i][j]) {
					belongVector[i] = j;
				} // of if
			} // of for j
		} // of for i
	}// of printNoiseDistribution

	/**
	 ********************** 
	 * Compute the predicted rating matrix.
	 ********************** 
	 */
	public void computePredictions() {
		predictions = tool.MatrixOpr.Matrix_Mult(factor_U, factor_V);
		System.out.println("The size of paraMatrix_Multi: "
				+ predictions.length + "*" + predictions[0].length);
		predictions = tool.MatrixOpr.Add_MatrixandNumber(predictions,
				data.meanOfRatings);

		// SimpleTool.printMatrix(paraMatrix_Multi);
		for (int i = 0; i < predictions.length; i++) {
			for (int j = 0; j < predictions[0].length; j++) {
				// if (data.testMatrix[i][j] < 1e-6)
				// continue;
				if (predictions[i][j] < 1) {
					// System.out.println("Value exceeds bound: "
					// + predictions[i][j]);
					predictions[i][j] = 1;
				} // of if
				if (predictions[i][j] > 5) {
					predictions[i][j] = 5;
				} // of if
			} // of for j
		} // of for i
	}// Of computePredictions

	/**
	 ************************** 
	 * @param args
	 * @throws IOException
	 ************************** 
	 */
	public static void main(String[] args) throws IOException {

		try{
			// Prepare data and preprocessing
			String tempPropertyFileName = new String("src/properties/Filmtrust.properties");
			DataInfo tempData = new DataInfo(tempPropertyFileName);
			tempData.computeAverageRating();
			tempData.computeDataVector();
			tempData.recomputeDataset();// subtract the average of training set
			tempData.generateRandomSubMatrix();
			tempData.setRandomNoiseDistribution();
			tempData.computeWeight();// For the first time. Round 0

			MLGMDN tempEm = new MLGMDN(tempData);
			// tempEm.setModel();

			tempEm.iterationEM(5);
			tempEm.printNoiseDistribution();
			tempData.recoverRatingMatrix();
//			System.out.println("Latent variable z_{ijk}:");
//			SimpleTool.printMatrix(tempEm.z);

			tempEm.computePredictions();
			System.out.print("llh: ");
			SimpleTool.printFloatArray(tempEm.llh);
			System.out.print("noiseWeight: ");
			SimpleTool.printFloatArray(tempEm.noiseWeight);
			System.out.println("The end.");
		}catch(Exception e){
			e.printStackTrace();
		}//of try
		
	}// of main
}// of class MLGMDN
