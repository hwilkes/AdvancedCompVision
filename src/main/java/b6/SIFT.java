package b6;

import java.util.Map;

import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;

public class SIFT {
	DoGSIFTEngine engine;
	
	private String[] classes;
	
	
	public SIFT(){
		engine = new DoGSIFTEngine();	
	}
	
	public void trainImages(String[] classes, Map<String,FImage> image){
		this.classes = classes;
		
		
	}
	
	public String classify(FImage image){
		return null;
	}
}
