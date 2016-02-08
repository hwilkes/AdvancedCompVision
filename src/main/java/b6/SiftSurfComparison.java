package b6;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;


public class SiftSurfComparison {
	
	public static void main(String args[]){
		try {
			SiftSurfComparison ssc = new SiftSurfComparison();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//TODO better file organisation
	private final String imageDir = "images/training/";
	
	private final float traningToTestRatio = 0.8f;
	
	private FImage[] testImages;
	private List<Annotated<FImage,String>> trainingImages;
	
	private String[] classes;
	
	private Random r;
	
	public SiftSurfComparison() throws IOException{
		List<String> cls = new ArrayList<String>();
		List<FImage> ti = new ArrayList<FImage>();
		
		r = new Random();
		trainingImages = new ArrayList<Annotated<FImage,String>>();
		
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
				File[] imageFiles = f.listFiles();
				FImage[] images = new FImage[imageFiles.length];
				
				for(int i = 0; i < imageFiles.length; i++){
					images[i] = ImageUtilities.readF(imageFiles[i]); 
				}
				
				String c = f.getName();
				for(FImage i : images){
					if(testNum != 0 && r.nextBoolean()){
						ti.add(i);
						testNum--;
					}
					else{
						trainingImages.add(new AnnotatedObject<FImage,String>(i,c));
					}
				}
			}
		}
		
		this.testImages = ti.toArray(new FImage[ti.size()]);
		this.classes = cls.toArray(new String[cls.size()]);
		
	}
	
	public void train(){

	}
}
