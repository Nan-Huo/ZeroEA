import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;

public class ZeroEA {

	public static void main(String[] args) throws IOException {
        
        //String mainDir = "./";
        //String[]dirs = { "KG1_fr","KG2_fr", "KG1_wd","KG2_wd", "KG1_yg", "KG2_yg", "KG1_zh", "KG2_zh", "KG1_ja", "KG2_ja", "KG1_fr2", "KG2_fr2", "KG1_de", "KG2_de"};
        //for(int i=0;i<dirs.length;i++){
        //	triangleEnu(mainDir, dirs[i]);
        	//allmotifs();
        //}
		
		String mainDir = args[0]+"/", dir = args[1];
		if(args.length==2 || (args.length==3&&args[2].equals("triangle")))
			triangleEnu(mainDir, dir);
		if(args.length==3&&args[2].equals("all"))
			allmotifs(mainDir, dir);
	}
	
	static void triangleEnu(String mainDir, String dir) throws IOException {
		graphReady g = new graphReady();
		g.normalize_root(mainDir+dir," ", 0, 1, 2, 0, -1, -1, false);
		long ti2 = System.currentTimeMillis();

		ti2 = System.currentTimeMillis();
		for(int i=1;i<g.graph.length;i++) {
			int s = i;
			int triangle =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int t = g.graph[s].get(j);
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem1 = g.graph[s].get(p);
					if(g.graph[t].contains(tem1)) {
						triangle ++;
					}
				}
			}
		}
		long ti3 = System.currentTimeMillis();
		System.out.println(dir+"\t"+(ti3-ti2));		
	}
	public static void allmotifs(String mainDir, String dir) throws IOException {
		//String mainDir = "C:\\Users\\Xiaodongli\\Downloads\\KGs\\";
		//String[]dirs= {"KG1_de","KG2_de","KG1_fr","KG2_fr"};
		//int[]edgeNums = {2215036, 1478552, 2698135, 2013777};
		FileOps fo = new FileOps();

		//String mainDir = args[0]+"/", dir = args[1];
		//String mainDir = "C:\\Users\\Xiaodongli\\Downloads\\KGs\\", dir = "KG1_zh";
		long ti = System.currentTimeMillis();
		BufferedReader a1 = fo.BRead(mainDir+dir);
		int edgeNum = 0;
		String ss = a1.readLine();
		while(ss!=null&&!ss.equals("")) {edgeNum++; ss=a1.readLine();}
		System.out.print(dir+" has \t"+edgeNum+"\tedges\t");
		a1.close();
	
		int[][]edges = new int[edgeNum][16];//s, t, original_s, original_t, wedge, triangle, fourstar, fourpath, tailedtriangle, fourcycle, dimond, fourclique, fivepath, fivestar, fivecycle, fiveclique
		BufferedReader a2 = fo.BRead(mainDir+dir);
		
		graphReady g = new graphReady();
		g.normalize_root(mainDir+dir," ", 0, 1, 2, 0, -1, -1, false);
		for(int i=0;i<edgeNum;i++) {
			String[]tem=a2.readLine().split(" ");
			int so = Integer.parseInt(tem[0]), s = g.map.get(tem[0]);
			int to = Integer.parseInt(tem[1]), t = g.map.get(tem[1]);
			edges[i][0] = s;
			edges[i][1] = t;
			edges[i][2] = so;
			edges[i][3] = to;	
		}
		a2.close();
		long ti2 = System.currentTimeMillis();
		System.out.println(g.nodeNum+"\tnodes\t");
		System.out.println(dir+" load time:\t"+((ti2-ti))+"\t milliseconds");
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int wedge =0;
			for(int j=0;j<g.graph[s].size();j++) {
				if(g.graph[s].get(j)==t)
					continue;
				if(!g.graph[t].contains(g.graph[s].get(j))) {
					wedge ++;
				}
			}
			for(int j=0;j<g.graph[t].size();j++) {
				if(g.graph[t].get(j)==s)
					continue;
				if(!g.graph[s].contains(g.graph[t].get(j))) {
					wedge ++;
				}
			}
			edges[i][4] = wedge;		
		}
		long ti3 = System.currentTimeMillis();
		System.out.println(dir+" wedge search time:\t"+(ti3-ti2)+"\t milliseconds");	

		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int triangle =0;
			for(int j=0;j<g.graph[s].size();j++) {
				if(g.graph[s].get(j)==t)
					continue;
				if(g.graph[t].contains(g.graph[s].get(j))) {
					triangle ++;
				}
			}
			edges[i][5] = triangle;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" triangle search time:\t"+(ti3-ti2)+"\t milliseconds");			

		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int threestar =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					if(!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)&&!g.graph[tem1].contains(tem2)) {
						threestar ++;
					}
				}
			}
			int tems = s; s = t; t = tems;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					if(!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)&&!g.graph[tem1].contains(tem2)) {
						threestar ++;
					}
				}
			}
			edges[i][6] = threestar;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" threestar search time:\t"+(ti3-ti2)+"\t milliseconds");	
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int threepath =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=0;p<g.graph[tem1].size();p++) {
					int tem2 = g.graph[tem1].get(p);
					if(tem2==s) continue;
					if(!g.graph[s].contains(tem2)&&!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)) {
						threepath ++;
					}
				}
				for(int p=0;p<g.graph[t].size();p++) {
					int tem2 = g.graph[t].get(p);
					if(tem2==s) continue;
					if(!g.graph[s].contains(tem2)&&!g.graph[t].contains(tem1)&&!g.graph[tem1].contains(tem2)) {
						threepath ++;
					}
				}
			}
			int tems = s; s = t; t = tems;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=0;p<g.graph[tem1].size();p++) {
					int tem2 = g.graph[tem1].get(p);
					if(tem2==s) continue;
					if(!g.graph[s].contains(tem2)&&!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)) {
						threepath ++;
					}
				}
			}
			edges[i][7] = threepath;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" threepath search time:\t"+(ti3-ti2)+"\t milliseconds");	
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int tailedtriangle =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t||!g.graph[t].contains(tem1)) continue;
				for(int p=0;p<g.graph[tem1].size();p++) {
					int tem2 = g.graph[tem1].get(p);
					if(tem2==s||tem2==t) continue;
					if(!g.graph[s].contains(tem2)&&!g.graph[t].contains(tem2)) {
						tailedtriangle ++;
					}
				}
			}
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					if(g.graph[tem1].contains(tem2)&&!g.graph[t].contains(tem2)&&!g.graph[t].contains(tem1)) {
						tailedtriangle ++;
					}
					if(g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)&&!g.graph[tem1].contains(tem2)) {
						tailedtriangle ++;
					}
					if(g.graph[t].contains(tem2)&&!g.graph[t].contains(tem1)&&!g.graph[tem2].contains(tem1)) {
						tailedtriangle ++;
					}
				}
			}
			int tems = s; s = t; t = tems;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					if(g.graph[tem1].contains(tem2)&&!g.graph[t].contains(tem2)&&!g.graph[t].contains(tem1)) {
						tailedtriangle ++;
					}
					if(g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)&&!g.graph[tem1].contains(tem2)) {
						tailedtriangle ++;
					}
					if(g.graph[t].contains(tem2)&&!g.graph[t].contains(tem1)&&!g.graph[tem2].contains(tem1)) {
						tailedtriangle ++;
					}
				}
			}
			edges[i][8] = tailedtriangle;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" tailedtriangle search time:\t"+(ti3-ti2)+"\t milliseconds");
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int fourcycle =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=0;p<g.graph[t].size();p++) {
					int tem2 = g.graph[t].get(p);
					if(tem2==s) continue;
					if(g.graph[tem1].contains(tem2)&&!g.graph[s].contains(tem1)&&!g.graph[t].contains(tem2)) {
						fourcycle ++;
					}
				}
			}
			edges[i][9] = fourcycle;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" fourcycle search time:\t"+(ti3-ti2)+"\t milliseconds");	
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int diamond =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					if(g.graph[t].contains(tem1)&&g.graph[t].contains(tem2)&&!g.graph[tem1].contains(tem2)) {
						diamond ++;
					}
					if(g.graph[t].contains(tem1)&&g.graph[tem1].contains(tem2)&&!g.graph[t].contains(tem2)) {
						diamond ++;
					}
					if(g.graph[t].contains(tem2)&&g.graph[tem1].contains(tem2)&&!g.graph[t].contains(tem1)) {
						diamond ++;
					}
				}
			}
			int tems = s; s = t; t = tems;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					if(g.graph[t].contains(tem1)&&g.graph[tem1].contains(tem2)&&!g.graph[t].contains(tem2)) {
						diamond ++;
					}
					if(g.graph[t].contains(tem2)&&g.graph[tem1].contains(tem2)&&!g.graph[t].contains(tem1)) {
						diamond ++;
					}
				}
			}
			edges[i][10] = diamond;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" diamond search time:\t"+(ti3-ti2)+"\t milliseconds");	
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int fourclique =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					if(g.graph[tem1].contains(tem2)&&g.graph[t].contains(tem1)&&g.graph[t].contains(tem2)) {
						fourclique ++;
					}
				}
			}
			edges[i][11] = fourclique;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" fourclique search time:\t"+(ti3-ti2)+"\t milliseconds");
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int fivepath =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=0;p<g.graph[tem1].size();p++) {
					int tem2 = g.graph[tem1].get(p);
					if(tem2==s) continue;
					for(int q=0;q<g.graph[tem2].size();q++) {
						int tem3 = g.graph[tem2].get(q);
						if(tem3==tem1) continue;
						if(!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)&&!g.graph[t].contains(tem3)&&!g.graph[s].contains(tem2)&&!g.graph[s].contains(tem3)&&!g.graph[tem1].contains(tem3)) {
							fivepath++;
						}
					}
					for(int q=0;q<g.graph[t].size();q++) {
						int tem3 = g.graph[t].get(q);
						if(tem3==s) continue;
						if(!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)&&!g.graph[s].contains(tem2)&&!g.graph[s].contains(tem3)&&!g.graph[tem1].contains(tem3)&&!g.graph[tem2].contains(tem3)) {
							fivepath++;
						}
					}	
				}
				for(int p=0;p<g.graph[t].size();p++) {
					int tem2 = g.graph[t].get(p);
					if(tem2==s) continue;
					for(int q=0;q<g.graph[tem2].size();q++) {
						int tem3 = g.graph[tem2].get(q);
						if(tem3==t) continue;
						if(!g.graph[s].contains(tem2)&&!g.graph[s].contains(tem3)&&!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem3)&&!g.graph[tem1].contains(tem2)&&!g.graph[tem1].contains(tem3)) {
							fivepath++;
						}
					}
				}
			}
			for(int j=0;j<g.graph[t].size();j++) {
				int tem1 = g.graph[t].get(j);
				if(tem1==s) continue;
				for(int p=0;p<g.graph[tem1].size();p++) {
					int tem2 = g.graph[tem1].get(p);
					if(tem2==t) continue;
					for(int q=0;q<g.graph[tem2].size();q++) {
						int tem3 = g.graph[tem2].get(q);
						if(tem3==tem1) continue;
						if(!g.graph[s].contains(tem1)&&!g.graph[s].contains(tem2)&&!g.graph[s].contains(tem3)&&!g.graph[t].contains(tem2)&&!g.graph[t].contains(tem3)&&!g.graph[tem1].contains(tem3)) {
							fivepath++;
						}
					}
				}
			}
			edges[i][12] = fivepath;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" fivepath search time:\t"+(ti3-ti2)+"\t milliseconds");
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int fivestar =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					for(int q=p+1;q<g.graph[s].size();q++) {
						int tem3 = g.graph[s].get(q);
						if(tem3==t) continue;
						if(!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem2)&&!g.graph[t].contains(tem3)&&!g.graph[tem1].contains(tem2)&&!g.graph[tem1].contains(tem3)&&!g.graph[tem2].contains(tem3)) {
							fivestar++;
						}
					}
				}
			}
			for(int j=0;j<g.graph[t].size();j++) {
				int tem1 = g.graph[t].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[t].size();p++) {
					int tem2 = g.graph[t].get(p);
					if(tem2==s) continue;
					for(int q=p+1;q<g.graph[t].size();q++) {
						int tem3 = g.graph[t].get(q);
						if(tem3==s) continue;
						if(!g.graph[s].contains(tem1)&&!g.graph[s].contains(tem2)&&!g.graph[s].contains(tem3)&&!g.graph[tem1].contains(tem2)&&!g.graph[tem1].contains(tem3)&&!g.graph[tem2].contains(tem3)) {
							fivestar++;
						}
					}
				}
			}
			edges[i][13] = fivestar;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" fivestar search time:\t"+(ti3-ti2)+"\t milliseconds");
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int fivecycle =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=0;p<g.graph[t].size();p++) {
					int tem2 = g.graph[t].get(p);
					if(tem2==s) continue;
					for(int q=0;q<g.graph[tem1].size();q++) {
						int tem3 = g.graph[tem1].get(q);
						if(tem3==s) continue;
						if(g.graph[tem2].contains(tem3)&&!g.graph[s].contains(tem2)&&!g.graph[s].contains(tem3)&&!g.graph[t].contains(tem1)&&!g.graph[t].contains(tem3)&&!g.graph[tem1].contains(tem3)) {
							fivecycle++;
						}
					}
				}
			}
			edges[i][14] = fivecycle;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" fivecycle search time:\t"+(ti3-ti2)+"\t milliseconds");
		
		ti2 = System.currentTimeMillis();
		for(int i=0;i<edgeNum;i++) {
			int s = edges[i][0], t = edges[i][1];
			//if(!g.graph[s].contains(t)) {fo.jout("Wrong: "+s +"\t"+t);break;}
			int fiveclique =0;
			for(int j=0;j<g.graph[s].size();j++) {
				int tem1 = g.graph[s].get(j);
				if(tem1==t) continue;
				for(int p=j+1;p<g.graph[s].size();p++) {
					int tem2 = g.graph[s].get(p);
					if(tem2==t) continue;
					for(int q=p+1;q<g.graph[s].size();q++) {
						int tem3 = g.graph[s].get(q);
						if(tem3==t) continue;
						if(g.graph[t].contains(tem1)&&g.graph[t].contains(tem2)&&g.graph[t].contains(tem3)&&g.graph[tem1].contains(tem2)&&g.graph[tem1].contains(tem3)&&g.graph[tem2].contains(tem3)) {
							fiveclique++;
						}
					}
				}
			}
			edges[i][15] = fiveclique;		
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" fiveclique search time:\t"+(ti3-ti2)+"\t milliseconds");
		
		ti2 = System.currentTimeMillis();
		BufferedWriter b = fo.BWriter(mainDir+dir+"new");
		for(int i=0;i<edgeNum;i++) {
			for(int j=2;j<16;j++) {
				b.write(edges[i][j]+" ");
			}
			b.write("\n");
		}
		ti3 = System.currentTimeMillis();
		System.out.println(dir+" write time:\t"+(ti3-ti2)+"\t milliseconds");
		
		b.flush();
		b.close();
	}
	

}
