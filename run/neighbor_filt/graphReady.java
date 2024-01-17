
import java.awt.List;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

//import graph_clustering.Ground_truth;

public class graphReady {
	/*
	 * Note: the graph IDs starts from 1 in ArrayList<Integer>graph[] and
	 * ArrayList<Double>w[]. ReadGraph - read the graph without nodeNum (read_path,
	 * separator, left, right, weight line, uselessLines, nodeNum, ifWeighted)
	 * readGraph - read the graph (read_path, separator, left, right, weight line,
	 * uselessLines, edgeNum, nodeNum, ifWeighted) graph - read the graph (
	 * normalize - make the IDs continues (read_path, separator, left, right, weight
	 * line, uselessLines, edgeNum, nodeNum, ifWeighted) getNodeNum - return nodeNum
	 * (read_path, separator, left, right, uselessLines, edgeNum) addRandomWeight -
	 * add random weights on the edges writeGraph - write graph into writeDir
	 * (writeDir, separator, ifWeighted) diameter() - return the diameter of the
	 * graph
	 * 
	 */

	public ArrayList<Integer> graph[]; // to store the uncertain graph into a list
	public ArrayList<Double> w[]; // Similar structure as graph, but a UGDiameter-size vector serve as the element
									// on each edge
	public int edgeNum;
	public int nodeNum;

	public void readGraph_root(String s, String sp, int l, int r, int wl, int startLine, int edgeNum, int nodeNum,
			boolean ifWeighted, boolean ifAdd1) throws IOException {

		if (nodeNum != -1 && edgeNum != -1) {
			this.nodeNum = nodeNum;
			this.edgeNum = edgeNum;
		} else {
			this.nodeNum = getNodeNum(s, sp, l, r, startLine);
			edgeNum = this.edgeNum;
			nodeNum = this.nodeNum;
		}

		graph = new ArrayList[nodeNum + 1];
		w = new ArrayList[nodeNum + 1];

		for (int i = 1; i < graph.length; i++) {
			graph[i] = new ArrayList<Integer>();
			w[i] = new ArrayList<Double>();
		}
		BufferedReader a = new BufferedReader(new FileReader(s));
		for (int i = 0; i < startLine; i++) {
			a.readLine();
		}
		for (int i = 0; i < edgeNum; i++) {
			String str = a.readLine();
			String[] tem = str.split(sp);
				
			int left = Integer.parseInt(tem[l]), right = Integer.parseInt(tem[r]);
			if(left > graph.length || right > graph.length) continue;
			if (ifAdd1) {
				left++;
				right++;
			}
			double weight = -1;

			if (!graph[left].contains(right)) {
				graph[left].add(right);
				if (ifWeighted) {
					weight = Double.parseDouble(tem[wl]);
					w[left].add(weight);
				} else {
					weight = 1.0;
					w[left].add(weight);
				}
			}
			if (!graph[right].contains(left)) {
				graph[right].add(left);
				if (ifWeighted) {
					weight = Double.parseDouble(tem[wl]);
					w[right].add(weight);
				} else {
					weight = 1.0;
					w[right].add(weight);
				}
			}

		}

		this.match = new String[nodeNum + 1];
		this.map = new HashMap();
		for (int i = 1; i <= nodeNum; i++) {
			this.match[i] = i + "";
			this.map.put(i + "", i);
		}

	}

	public void readGraph(String s, String sp, int l, int r, int wl, int startLine, boolean ifWeighted)
			throws IOException {
		int nodeNum = -1, edgeNum = -1;
		boolean ifAdd1 = false;
		readGraph_root(s, sp, l, r, wl, startLine, edgeNum, nodeNum, ifWeighted, ifAdd1);
		/*
		 * this.nodeNum=getNodeNum(s,sp,l,r,startLine); graph = new
		 * ArrayList[nodeNum+1]; w = new ArrayList[nodeNum+1];
		 * 
		 * for(int i=1;i<graph.length;i++){ graph[i]=new ArrayList<Integer>();
		 * if(ifWeighted)w[i]= new ArrayList<Double>(); } BufferedReader a = new
		 * BufferedReader(new FileReader(s)); for(int i=0;i<startLine;i++){
		 * a.readLine(); } for(int i=0;i<edgeNum;i++){ String[]tem =
		 * a.readLine().split(sp); int left = Integer.parseInt(tem[l]), right =
		 * Integer.parseInt(tem[r]); double weight=-1;
		 * 
		 * if(!graph[left].contains(right)){ graph[left].add(right); if(ifWeighted){
		 * weight=Double.parseDouble(tem[wl]); w[left].add(weight); } else{ weight=1.0;
		 * w[left].add(weight); } } if(!graph[right].contains(left)){
		 * graph[right].add(left); if(ifWeighted){ weight=Double.parseDouble(tem[wl]);
		 * w[right].add(weight); } else{ weight=1.0; w[right].add(weight); } }
		 * 
		 * }
		 */
	}

	public void readGraph(String s, String sp, int l, int r, int wl, int startLine, boolean ifWeighted, boolean ifAdd1)
			throws IOException {
		int nodeNum = -1, edgeNum = -1;
		readGraph_root(s, sp, l, r, wl, startLine, edgeNum, nodeNum, ifWeighted, ifAdd1);
	}

	public void readGraph(String s, String sp, int l, int r, int startLine, boolean ifAdd1) throws IOException {
		int nodeNum = -1, edgeNum = -1, wl = -1;
		boolean ifWeighted = false;
		readGraph_root(s, sp, l, r, wl, startLine, edgeNum, nodeNum, ifWeighted, ifAdd1);
	}

	public void readGraph(String s, String sp, boolean ifAdd1) throws IOException {
		int l = 0, r = 1, startLine = 0;
		int nodeNum = -1, edgeNum = -1, wl = -1;
		boolean ifWeighted = false;
		readGraph_root(s, sp, l, r, wl, startLine, edgeNum, nodeNum, ifWeighted, ifAdd1);
	}

	public void readGraph(String s, String sp, int l, int r, int wl, int startLine, int edgeNum, int nodeNum,
			boolean ifWeighted) throws IOException {

		boolean ifAdd1 = false;
		readGraph_root(s, sp, l, r, wl, startLine, edgeNum, nodeNum, ifWeighted, ifAdd1);

		/*
		 * this.edgeNum=edgeNum; graph = new ArrayList[nodeNum+1]; w = new
		 * ArrayList[nodeNum+1];
		 * 
		 * for(int i=1;i<graph.length;i++){ graph[i]=new ArrayList<Integer>(); }
		 * BufferedReader a = new BufferedReader(new FileReader(s)); for(int
		 * i=0;i<startLine;i++){ a.readLine(); } for(int i=0;i<edgeNum;i++){
		 * 
		 * String[]tem = a.readLine().split(sp); int left = Integer.parseInt(tem[l]),
		 * right = Integer.parseInt(tem[r]); double weight=-1;
		 * 
		 * if(Double.parseDouble(tem[wl])==0)continue;
		 * 
		 * if(!graph[left].contains(right)){ graph[left].add(right); if(ifWeighted){
		 * weight=Double.parseDouble(tem[wl]); w[left].add(weight); } else{ weight=1.0;
		 * w[left].add(weight); } } if(!graph[right].contains(left)){
		 * graph[right].add(left); if(ifWeighted){ weight=Double.parseDouble(tem[wl]);
		 * w[right].add(weight); } else{ weight=1.0; w[right].add(weight); } }
		 * 
		 * }
		 */

	}

	public void readGraph(String s, String sp, int l, int r, int wl, int startLine, int edgeNum, int nodeNum,
			boolean ifWeighted, boolean ifAdd1) throws IOException {
		readGraph_root(s, sp, l, r, wl, startLine, edgeNum, nodeNum, ifWeighted, ifAdd1);
	}

	public String[] match = null;
	public HashMap<String, Integer> map = null;

	public void normalize_root(String s, String sp, int nodeLine1, int nodeLine2, int WeightedLine, int startLine,
			int edgeNum, int nodeNum, boolean weighted) throws IOException {
		// to normalize the graph into the one with continuous nodes ID starting from 1
		// unweighted graph will be stored into graph with weight=1

		if (nodeNum != -1 && edgeNum != -1) {
			this.nodeNum = nodeNum;
			this.edgeNum = edgeNum;
		} else {
			this.nodeNum = getNodeNum(s, sp, nodeLine1, nodeLine2, startLine);
			edgeNum = this.edgeNum;
			nodeNum = this.nodeNum;
		}

		graph = new ArrayList[nodeNum + 1];
		w = new ArrayList[nodeNum + 1];
		match = new String[nodeNum + 1];

		for (int i = 1; i < graph.length; i++) {
			graph[i] = new ArrayList<Integer>();
			w[i] = new ArrayList<Double>();
		}

		BufferedReader a = new BufferedReader(new FileReader(s));

		for (int i = 0; i < startLine; i++)
			a.readLine();

		map = new HashMap<String, Integer>();

		int mapCounter = 0;

		if (weighted) {

			for (int i = 0; i < edgeNum; i++) {

				String[] tem = a.readLine().split(sp);
				int left = 0, right = 0;
				// to map the nodes into numbers
				if (!map.containsKey(tem[nodeLine1])) {
					mapCounter++;
					map.put(tem[nodeLine1], mapCounter);
					left = mapCounter;
					match[left] = (tem[nodeLine1]);
				} else {
					left = map.get(tem[nodeLine1]);
				}

				if (!map.containsKey(tem[nodeLine2])) {
					mapCounter++;
					map.put(tem[nodeLine2], mapCounter);
					right = mapCounter;
					match[right] = (tem[nodeLine2]);
				} else {
					right = map.get(tem[nodeLine2]);
				}
				double weight = -1;
				weight = Double.parseDouble(tem[WeightedLine]);

				// graph[left][right]=weight;
				// graph[right][left]=weight;

				if (!graph[left].contains(right)) {
					graph[left].add(right);
					w[left].add(weight);
				}
				if (!graph[right].contains(left)) {
					graph[right].add(left);
					w[right].add(weight);
				}
			}
		} else {
			for (int i = 0; i < edgeNum; i++) {

				String[] tem = a.readLine().split(sp);
				int left = 0, right = 0;
				// to map the nodes into numbers
				if (!map.containsKey(tem[nodeLine1])) {
					mapCounter++;
					map.put(tem[nodeLine1], mapCounter);
					left = mapCounter;
					match[left] = (tem[nodeLine1]);
				} else {
					left = map.get(tem[nodeLine1]);

				}

				if (!map.containsKey(tem[nodeLine2])) {
					mapCounter++;
					map.put(tem[nodeLine2], mapCounter);
					right = mapCounter;
					match[right] = (tem[nodeLine2]);
				} else {
					right = map.get(tem[nodeLine2]);
				}

				if (!graph[left].contains(right)) {
					graph[left].add(right);

				}
				if (!graph[right].contains(left)) {
					graph[right].add(left);

				}

			}
		}

	}

	public void normalize(String s, String sp, int nodeLine1, int nodeLine2, int WeightedLine, int startLine,
			int edgeNum, int nodeNum, boolean weighted) throws IOException {
		normalize_root(s, sp, nodeLine1, nodeLine2, WeightedLine, startLine, edgeNum, nodeNum, weighted);
	}

	public void normalize(String s, String sp, int nodeLine1, int nodeLine2, int WeightedLine, int startLine,
			boolean weighted) throws IOException {
		int nodeNum = -1, edgeNum = -1;
		normalize_root(s, sp, nodeLine1, nodeLine2, WeightedLine, startLine, edgeNum, nodeNum, weighted);
		/*
		 * this.nodeNum=getNodeNum(s,sp,nodeLine1,nodeLine2,startLine); graph = new
		 * ArrayList[nodeNum+1]; w = new ArrayList[nodeNum+1];
		 * 
		 * for(int i=1;i<graph.length;i++){ graph[i]=new ArrayList<Integer>(); w[i]=new
		 * ArrayList<Double>(); } BufferedReader a = new BufferedReader(new
		 * FileReader(s)); for(int i=0;i<startLine;i++){ a.readLine(); }
		 * 
		 * match = new String[nodeNum+1]; map = new HashMap<String, Integer>(); int
		 * mapCounter = 0;
		 * 
		 * for(int i=0;i<edgeNum;i++){ String[]tem=null; try{ tem =
		 * a.readLine().split(sp); }catch(Exception e){ System.out.println(); } int left
		 * = 0, right = 0; // to map the nodes into numbers
		 * if(!map.containsKey(tem[nodeLine1])){ mapCounter++; map.put(tem[nodeLine1],
		 * mapCounter); left = mapCounter; match[left] = (tem[nodeLine1]); }else{ left =
		 * map.get(tem[nodeLine1]); }
		 * 
		 * 
		 * if(!map.containsKey(tem[nodeLine2])){ mapCounter++; map.put(tem[nodeLine2],
		 * mapCounter); right = mapCounter; match[right] = (tem[nodeLine2]); }else{
		 * right = map.get(tem[nodeLine2]); } double weight= -1; if(weighted) weight =
		 * Double.parseDouble(tem[WeightedLine]); else weight=1;
		 * //graph[left][right]=weight; //graph[right][left]=weight;
		 * 
		 * if(!graph[left].contains(right)){ graph[left].add(right);
		 * w[left].add(weight); } if(!graph[right].contains(left)){
		 * graph[right].add(left); w[right].add(weight); }
		 * 
		 * }
		 */

	}

	public int getNodeNum(String s, String sp, int line1, int line2, int uselessLines) throws IOException {
		HashSet<String> h = new HashSet<String>();
		BufferedReader a = new BufferedReader(new FileReader(s));
		for (int i = 0; i < uselessLines; i++) {
			String str = a.readLine();
			if(str.contains("um")) {
				//nodenum: x, edgenum: y
				String[]tem = str.split("	");
				if(tem[i].contains(":")) {
					String ts = tem[1].replaceAll("#", "").split(":")[1].replaceAll(" ", "");
					edgeNum = Integer.parseInt(ts);
					return Integer.parseInt(tem[0].split(":")[1].replace(" ", ""));
				}else {
					edgeNum = Integer.parseInt(tem[1].replace(" ", ""));
					return Integer.parseInt(tem[0].replace(" ", ""));
				}
				
			}
			}
		String[] tem = null;
		String sa = a.readLine();
		edgeNum = 0;
		while (sa != null && !sa.equals("")) {
			tem = sa.split(sp);
			h.add(tem[line1]);
			try {
				h.add(tem[line2]);
			} catch (Exception e) {
				System.out.println(sa);
			}
			sa = a.readLine();
			edgeNum++;
			//if(edgeNum%(1397278/100)==0)
			//	System.out.print(edgeNum/(1397278/100)+"->");
		}
		return h.size();
	}

	// boolean reached[];
	// int reachNum;
	public int[] compSig;// the component-id of each node
	public int components = -1;// the number of components
	public int no = -1;// one member in the main component



	public double getDegree() {
		double d = 0;
		for (int i = 1; i < graph.length; i++) {
			d += graph[i].size();
		}
		d /= (graph.length - 1);
		return d;
	}
	
	public ArrayList<Integer>[] getSubgraph (ArrayList<Integer> S){
		//map: S - id of S +1
		ArrayList<Integer>[] subgraph = new ArrayList[S.size()+1];
		for(int i=0;i<S.size();i++) {
			int cn = S.get(i);
			ArrayList<Integer>cneis = graph[cn];
			ArrayList<Integer>cnew = new ArrayList();
			for(int j=0;j<cneis.size();j++) {
				if(S.contains(cneis.get(j))) {
					//edge: cn - cneis.get(j)
					cnew.add(S.indexOf(cneis.get(j))+1);
				}
			}
			subgraph[i+1] = cnew;
		}
		return subgraph;
	} 

}
