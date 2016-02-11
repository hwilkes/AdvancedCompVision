package b6;

import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.ByteFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.ml.annotation.Annotated;
import org.openimaj.ml.annotation.AnnotatedObject;
import org.openimaj.ml.annotation.ScoredAnnotation;


public class SiftSurfComparison {
	
	public static void main(String args[]){
		try {
			SiftSurfComparison ssc = new SiftSurfComparison();
			
			ssc.train();
			ssc.test();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//TODO better file organisation
	private final String imageTrainingDir = "images/training/";
	private final String imageTestingDir = "images/testing/";
	
	private final float traningToTestRatio = 0.8f;
	
	private FImage[] testImages;
	private Map<String,FImage[]> trainingImages;
	
	private String[] classes;
	
	private Random r;
	
	SIFT dsift;
	
	public SiftSurfComparison() throws IOException{
		dsift = new SIFT();
		
		List<String> cls = new ArrayList<String>();
		List<FImage> ti = new ArrayList<FImage>();
		
		r = new Random();
		trainingImages = new HashMap<String,FImage[]>();
		
		File trainingdir = new File(imageTrainingDir);
		File testingdir = new File(imageTestingDir);
		System.out.println(trainingdir.getAbsolutePath());
		File[] classes = trainingdir.listFiles();
		
		for(File fi : testingdir.listFiles()){
			System.out.println(fi.getAbsolutePath());
			FImage toAdd = ImageUtilities.readF(fi); 
			ti.add(toAdd);
			
		}
		
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
					if(imageFiles[i].getName().endsWith(".jpg"))
					{
						images[i] = ImageUtilities.readF(imageFiles[i]); 
					}
				}
				
				String c = f.getName();
				for(FImage i : images){
					trainingImages.put(c, images);
				}
			}
		}
		
		this.testImages = ti.toArray(new FImage[ti.size()]);
		this.classes = cls.toArray(new String[cls.size()]);
		
	}
	
	public void train(){
		System.out.println("TRAINING");
		dsift.trainImages(this.classes, this.trainingImages);
		System.out.println("TRAINED");
	}
	
	public void test(){
		System.out.println("TESTING");
		List<List<ScoredAnnotation<String>>> results = new ArrayList<List<ScoredAnnotation<String>>>();
		int breakpoint = 0;
		for(FImage testImage : this.testImages){
			List<ScoredAnnotation<String>> annotations = dsift.classify(testImage);
			System.out.println("Classifying "+testImage.toString());
			results.add(annotations);
//			if(breakpoint == 100){
//				break;
//			}
//			else{
//				breakpoint++;
//			}
		}
		System.out.println(results.size());
		String output = "";
		int imageNum = 0;
		for(List<ScoredAnnotation<String>> cs : results){
			output = output + imageNum + ".jpg , ";
			for(ScoredAnnotation<String> anno : cs){
				
				output = output + anno.annotation+" , "+anno.confidence+" , ";
			}
			output = output.substring(0, output.length()-3);
			
			output = output+"\n";
			imageNum++;
		}
		
		try {
			File file = new File("output.txt");
			FileWriter fileWriter = new FileWriter(file);
			fileWriter.write(output);
			fileWriter.flush();
			fileWriter.close();
			System.out.println("OUTPUT CREATED");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
