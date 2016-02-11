package b6;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat4;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.list.MemoryLocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.analysis.pyramid.gaussian.GaussianOctave;
import org.openimaj.image.analysis.pyramid.gaussian.GaussianPyramid;
import org.openimaj.image.feature.local.descriptor.gradient.SIFTFeatureProvider;
import org.openimaj.image.feature.local.detector.dog.collector.Collector;
import org.openimaj.image.feature.local.detector.dog.collector.OctaveKeypointCollector;
import org.openimaj.image.feature.local.detector.dog.extractor.DominantOrientationExtractor;
import org.openimaj.image.feature.local.detector.dog.extractor.GradientFeatureExtractor;
import org.openimaj.image.feature.local.detector.dog.extractor.OrientationHistogramExtractor;
import org.openimaj.image.feature.local.detector.dog.pyramid.DoGOctaveExtremaFinder;
import org.openimaj.image.feature.local.detector.pyramid.BasicOctaveExtremaFinder;
import org.openimaj.image.feature.local.detector.pyramid.OctaveInterestPointFinder;
import org.openimaj.image.feature.local.engine.DoGSIFTEngineOptions;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.keypoints.InterestPointKeypoint;
import org.openimaj.image.feature.local.keypoints.Keypoint;
//import org.openimaj.image.feature.local.keypoints.Keypoint;

public class SURFEngine implements Engine<Keypoint,FImage> {
	
	private FeatureDetector fd;
	private DescriptorExtractor surfExtractor;
	
	public SURFEngine(){
		fd = FeatureDetector.create(FeatureDetector.SURF);
		surfExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
	}
	
	
	@Override
	public LocalFeatureList<Keypoint> findFeatures(FImage image) {
		//Mat matimage = new Mat(image.pixels.length,image.pixels[0].length,CvType.CV_8U);
		
		
		
//		for(int x = 0; x < image.pixels.length; x++){
//			for(int y = 0; y < image.pixels[0].length; y++){
//				matimage.put(x, y, image.pixels[x][y]);
//			}
//		}
		
		BufferedImage bimg = ImageUtilities.createBufferedImage(image);
		byte[] pixelsfalt = ((DataBufferByte) bimg.getRaster().getDataBuffer()).getData();
		
		Mat matimage = new Mat(bimg.getWidth(),bimg.getHeight(),CvType.CV_8UC(1));
		matimage.put(0, 0, pixelsfalt);
		
		MatOfKeyPoint keypoints = new MatOfKeyPoint();
		Mat descriptors = new Mat();
		
		fd.detect(matimage, keypoints);
		surfExtractor.compute(matimage, keypoints, descriptors);
		//this is wrong as far as I know
		float [] desc = new float[(int) (descriptors.total() * descriptors.channels())];
		descriptors.get(0,0,desc);
		List<byte[]> descs = new ArrayList<byte[]>();
		
		for(int i = 0; i < descriptors.rows(); i++){
			float[] d = new float[descriptors.cols()];
			descriptors.get(i, 0, d);
			byte[] b = new byte[d.length*4];
			int index = 0;
			for(int y = 0; y < d.length; y++){
				byte[] temp = float2ByteArray(d[y]);
				for(byte bt : temp){
					b[index] = bt;
					index++;
				}
			}
			
			descs.add(b);
		}

		LocalFeatureList<Keypoint> toReturn = new MemoryLocalFeatureList<Keypoint>();
		org.opencv.features2d.KeyPoint[] matkeys = keypoints.toArray();
	
		
		for(int i = 0; i < matkeys.length; i++){
			org.opencv.features2d.KeyPoint k = matkeys[i];
			Keypoint toAdd = new Keypoint();
			toAdd.x = (float) k.pt.x;
			toAdd.y = (float) k.pt.y;
			
			toAdd.scale = k.size;
			toAdd.ori = (float) Math.toRadians(k.angle);
			//so is this
			toAdd.ivec = descs.get(i);
			toReturn.add(toAdd);
		}
		
		return toReturn;
	}
	
	//http://stackoverflow.com/questions/14619653/converting-a-float-to-a-byte-array-and-vice-versa-in-java
	private byte [] float2ByteArray (float value)
	{  
	     return ByteBuffer.allocate(4).putFloat(value).array();
	}
	
}
