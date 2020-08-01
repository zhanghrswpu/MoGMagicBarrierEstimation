package datacleaning;

import java.io.File;
import java.io.FileWriter;
import java.io.RandomAccessFile;

/**
 * 
 * @author think
 *
 */
public class DataPretreatment {
	/**
	 * 
	 * @param paraFileName
	 * @throws Exception
	 */
	public DataPretreatment() throws Exception {
		String tempReader = "data/jester/jester-data-2.txt";
		String tempWrite = "data/jester/jester-data-2new.txt";
		
		buildNewData(tempReader, tempWrite);
	}// Of the first constructor

	/**
	 * 
	 * @param paraFileName
	 * @throws Exception
	 */
	public void buildNewData(String paraReader, String paraWriter) throws Exception{
		File tempFile = null;
		String tempString = null;
		String[] tempStrArray = null;

		// Compute values of arrays
		tempFile = new File(paraReader);
		if (!tempFile.exists()) {
			System.out.println("File is not exist!");
			return;
		}// Of if

		RandomAccessFile tempRanFile = new RandomAccessFile(tempFile, "r");
		// 读文件的起始位置
		int tempBeginIndex = 0;
		// 将读文件的开始位置移到beginIndex位置。
		tempRanFile.seek(tempBeginIndex);

		// Step 1. count the item degree
		int tempUserIndex = 0;
		int tempItemIndex = 0;
		double tempRating = 0;
		
		FileWriter fw = new FileWriter(new File(paraWriter));
		while ((tempString = tempRanFile.readLine()) != null) {
			tempStrArray = tempString.split("	");
			tempUserIndex ++;
				
			for(int i = 1; i < tempStrArray.length; i ++){
				tempItemIndex = i;
				tempRating = Double.parseDouble(tempStrArray[i]);
				if(tempRating != 99){
					fw.write(tempUserIndex + "," + tempItemIndex + "," + tempRating + "\r\n");
					fw.flush();
					System.out.println(tempUserIndex + "," + tempItemIndex + "," + tempRating);
				}//of if
			}//of for i		
		}// Of while
		System.out.println(tempUserIndex);
		tempRanFile.close();
		fw.close();
	}// Of buildRatingModel

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			DataPretreatment tempSoc = new DataPretreatment();
		} catch (Exception ee) {
			ee.printStackTrace();
		} // Of try

	}// Of main
}// Of Class SocialDataModel
