
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.security.MessageDigest;

public class FileOps {
	
	public BufferedReader BRead(String path) throws FileNotFoundException {
		BufferedReader a = new BufferedReader(new FileReader(path));
		return a;
	}
	public BufferedWriter BWriter(String path) throws IOException {
		BufferedWriter b = new BufferedWriter(new FileWriter(path));
		return b;
	}
	public ArrayList<String> getFiles(String path) {
	    ArrayList<String> files = new ArrayList<String>();
	    File file = new File(path);
	    File[] tempList = file.listFiles();
	    for (int i = 0; i < tempList.length; i++) {
	        if (tempList[i].isFile()) {
	            files.add(tempList[i].toString());
	        }
	    }
	    return files;
	}
	// ��ȡһ���ļ����������ļ������ļ����µ������ļ�����·��
	public ArrayList<String> getFilesAll(String filePath) {
		ArrayList<String>res = new ArrayList();
		File f = new File(filePath);
		File[] files = f.listFiles(); // �õ�f�ļ�������������ļ���
		//ArrayList<File> list = new ArrayList<File>();
		if(files!=null&&files.length>0)
		for (File file : files) {
			if (file.isDirectory()) {
				// ��ε�ǰ·�����ļ��У���ѭ����ȡ����ļ����µ������ļ�
				res.addAll(getFilesAll(file.getAbsolutePath()));
			} else {
				//list.add(file);
				res.add(file.getName());
			}
		}
		return res;

	}
	public ArrayList<File> getFilesAllRaw(String filePath) {
		ArrayList<File>res = new ArrayList();
		File f = new File(filePath);
		File[] files = f.listFiles(); 
		//ArrayList<File> list = new ArrayList<File>();
		if(files!=null&&files.length>0)
		for (File file : files) {
			if (file.isDirectory()) {
				res.addAll(getFilesAllRaw(file.getAbsolutePath()));
			} else {
				//list.add(file);
				res.add(file);
			}
		}
		return res;

	}
	public void move(String dir, String name, String newdir, String newname) {
		File folder = new File(newdir);
		if(!folder.exists()){
			folder.mkdirs();
		}
		File file = new File(dir+File.separator+name), newfile = new File(newdir+File.separator+newname);
		try{
			if(!file.renameTo(newfile)){
				System.out.println("Failed to remove!");
			}
		}catch(Exception e){
			System.out.println("Error when removing!");
		}
	}
	public void copy(String sourceDir, String destDir) throws IOException {
		File source = new File(sourceDir);
		File dest = new File(destDir);
	    InputStream input = null;    
	    OutputStream output = null;    
	    try {
	           input = new FileInputStream(source);
	           output = new FileOutputStream(dest);        
	           byte[] buf = new byte[1024];        
	           int bytesRead;        
	           while ((bytesRead = input.read(buf)) > 0) {
	               output.write(buf, 0, bytesRead);
	           }
	    } finally {
	        input.close();
	        output.close();
	    }
	}
	public boolean isSameFile (String oldName, String newName) {
		File oldFile = new File(oldName);
		File newFile = new File(newName);
		String oldFileMd5 = getFileMD5(oldFile);
		String newFileMd5 = getFileMD5(newFile);
		if (oldFileMd5 == null || newFileMd5 == null) {
			return false;
			}
		return (oldFileMd5.equals(newFileMd5));
	}
	
	private static String getFileMD5(File file) {
		if (!file.isFile()) {
			return null;
		}
		MessageDigest digest = null;
		FileInputStream inStream = null;
		byte buffer[] = new byte[1024];
		int len;
		try {
		digest = MessageDigest.getInstance("MD5");
		inStream = new FileInputStream(file);
		while ((len = inStream.read(buffer, 0, 1024)) != -1) {
		digest.update(buffer, 0, len);
		}
		inStream.close();
		} catch (Exception e) {
		e.printStackTrace();
		return null;
		}
		BigInteger bigInt = new BigInteger(1, digest.digest());
		return bigInt.toString(16);
	}
	
	private static void rename(String originf, String newf) {
		//System.out.println(originf+"\t"+newf);
		//if(newf.length()>1)
		//	return;
		
		String filePath = originf;
	    try {
	    	File src = new File(filePath);
	        filePath = newf;
	        File des = new File(filePath);
	        if (des.exists()) {
	        	boolean res = des.delete();
	            if (!res) {
	            	//System.out.println(getlf(originf)+"\t\t"+getlf(newf));
	            	System.out.println("Failed to delete file");
	            	}
	            }
	            if (!src.renameTo(des)) {
	            	//System.out.println(getlf(originf)+"\t\t"+getlf(newf));
	                System.out.println("Failed to renameTo file");
	            }
	        } catch (Exception e) {
	        	//System.out.println(getlf(originf)+"\t\t"+getlf(newf));
	            System.out.println(e.getMessage());
	        }
	    }
	
	private boolean ArrayContain(ArrayList<String> p, String s) {
		for(int i=0;i<p.size();i++){
			if(p.get(i).contains(s))
				return true;
		}
		return false;
	}
	
	public long getFileSizeMegaBytes(File file) {
		return (long)((double) file.length() / (1024 * 1024));
	}
	
	public long getFileSizeKiloBytes(File file) {
		return (long)((double) file.length() / 1024);
	}

	public long getFileSizeBytes(File file) {
		return file.length();
	}
	
	public static boolean isNumeric(String str) {
        for (int i = 0; i < str.length(); i++) {
            //System.out.println(str.charAt(i));
            if (!Character.isDigit(str.charAt(i))) {
                return false;
            }
        }
        return true;
    }
	public void jout(String s) throws IOException {
		System.out.print(s);
	}
	public void jout(int[]s) throws IOException {
		for(int i=0;i<s.length;i++)
			System.out.print(s[i]+" ");
		System.out.println();
	}	
	public void jout(String s, BufferedWriter b) throws IOException {
		b.write(s);
	}
	public void jout(String s, BufferedWriter b, int n) throws IOException {
		switch(n) {
		case 1:
			System.out.print(s);
			break;
		case 2:
			b.write(s);
			break;
		}
	}

	int[] getTimeFormat(FileTime ft) {
		int[] res = new int[2];
		String ba = ft.toString();
		System.out.print(ba+"\t");
		//ba = ba.replaceAll("Z", "");
		String[]bas = ba.split("T");
		bas[0] = bas[0].replaceAll("-", "");
		res[0]=Integer.parseInt(bas[0]);
		
		String[]tem=bas[1].split("[.]")[0].split(":");
		res[1] = Integer.parseInt(tem[0]+tem[1]);
		return res;
	}
	public String fill0Front(String s, int bit) {
		String res = "";
		for(int i=0;i<bit-s.length();i++) {
			res+="0";
		}
		res+=s;
		return res;
	}


	
}
