package b6;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class SiftSurfComparison {
	
	public static void main(String args[]){
		SiftSurfComparison ssc = new SiftSurfComparison();
	}
	
	//TODO better file organisation
	private final String imageDir = "images/training/";
	
	private final float traningToTestRatio = 0.8f;
	
	private File[] testImages;
	private Map<File,String> trainingImages;
	
	private String[] classes;
	
	private Random r;
	
	public SiftSurfComparison(){
		List<String> cls = new ArrayList<String>();
		List<File> ti = new ArrayList<File>();
		
		r = new Random();
		trainingImages = new HashMap<File,String>();
		
		File dir = new File(imageDir);
		
		File[] classes = dir.listFiles();
		
		
		
		int numImages = 0;
		
		for(File f : classes){
			if(f.isDirectory()){
				String imgclass = f.getName();
				cls.add(imgclass);
				numImages += f.listFiles().length;
			}
		}
		System.out.println(numImages);
		int trainNum = (int) (numImages - (numImages*traningToTestRatio));
		int testNum = numImages - trainNum;
		
		for(File f : classes){
			if(f.isDirectory()){
				File[] images = f.listFiles();
				String c = f.getName();
				for(File i : images){
					if(testNum != 0 && r.nextBoolean()){
						ti.add(i);
						testNum--;
					}
					else{
						trainingImages.put(i, c);
					}
				}
			}
		}
		
		this.testImages = ti.toArray(new File[ti.size()]);
		this.classes = cls.toArray(new String[cls.size()]);
		
	}
	
	public void train(){

	}
}
