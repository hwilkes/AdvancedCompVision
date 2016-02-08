package b6;

import java.util.Map;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;

public class SIFT {
	DoGSIFTEngine engine;
	
	private String[] classes;
	
	
	public SIFT(){
		engine = new DoGSIFTEngine();	
	}
	
	public void trainImages(String[] classes, Map<String,FImage> image){
		this.classes = classes;
		
		//LocalFeatureList<Keypoint> queryKeypoints = engine.findFeatures(query.flatten());
	}
	
	public String classify(FImage image){
		
		LocalFeatureList<Keypoint> imageKeypoints = engine.findFeatures(image);
		return null;
	}
}
