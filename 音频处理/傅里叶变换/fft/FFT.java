package fft;

import java.util.Arrays;

public class FFT {

	boolean inverse = false;
	
	Complex omega(int N,int k) {
		if(!inverse) {
			return new Complex(Math.cos(-2*Math.PI/N*k),Math.sin(-2*Math.PI/N*k));
		}
		return new Complex(Math.cos(-2*Math.PI/N*k),Math.sin(-2*Math.PI/N*k)).conj();
	}
	
	public Complex [] DFT_slow(Complex [] x) {
		//System.out.println("DFT_slow");
		int N = x.length;
		Complex [] result = new Complex[N];
		for(int k=0;k<N;k++) {
			result[k] = new Complex();
			for(int n=0;n<N;n++) {
				result[k]=result[k].add(x[n].mul(omega(N,k,n)));
				
			}
			//System.out.println(result[k].toString());
		}
		return result;
	}
	
	private Complex omega(int N, int k, int n) {
		// TODO Auto-generated method stub
		return new Complex(Math.cos(-2*Math.PI/N*k*n),Math.sin(-2*Math.PI/N*k*n));
	}

	//递归实现fft算法
	public Complex [] fft_recurs(Complex [] x) {
		int N = x.length;
//		if(N % 2 != 0) {
//			return null;
//		}else if(N==32) {
//			return DFT_slow(x);
		if(N==1) {
			return x;
		}else {
			Complex [] X_even = new Complex[N/2];
			Complex [] X_odd = new Complex[N/2];
			int M = N/2;
			// 按照系数奇偶划分为两半
			for(int i=0;i<M;++i) {
				X_even[i] = x[i*2];
				X_odd[i]=x[i*2+1];
			}
			X_even = fft_recurs(X_even);
			X_odd = fft_recurs(X_odd);
			
			Complex [] result = new Complex[N];
			for(int i=0;i<M;i++) {
				result[i] = new Complex();
				result[i+M]= new Complex();
				result[i] = X_even[i].add(omega(N,i).mul(X_odd[i]));
				result[i+M] = X_even[i].add(omega(N,i+M).mul(X_odd[i]));
			}
			return result;
		}
	}
	
	//Cooley-Tukey FFT
	public Complex [] fft(Complex [] x) {
		int N = x.length;
		if((Math.log((double)N)/Math.log(((double)2)))%1>0) {
			return null;
		}
		Complex [] omega = new Complex[N];
		for(int k=0;k<N;k++) {
			omega[k] = new Complex(Math.cos(-2 * Math.PI / N * k), Math.sin(-2 * Math.PI / N * k));
		}
		int k = 0;
		while((1<<k)<N) k++;
		//调整数据位置
		// egg: i=3, 011->110=000|100|010 
		for(int i=0;i<N;i++) {
			int t = 0;
			for(int j=0;j<k;j++) {
				if((i & (1 << j))>0) { //遍历二进制每一位
					t |=(1<<(k-j-1));  //如果二进制该位上值为1，则将对称位置设为1
				}
			}
			if(i<t) {
				Complex tmp = new Complex();
				tmp = x[i];
				x[i] = x[t];
				x[t] = tmp;
			}
		}
		//合并数据
		for(int l=2;l<=N;l*=2) {
			int m = l/2;
			for(int p=0;p!=N;p+=l) {
				for(int i=0;i<m;i++) {
					Complex t = new Complex();
					t = omega[N/l*i].mul(x[p+m+i]);
					x[p+m+i] = x[p+i].sub(t);
					x[p+i] = x[p+i].add(t);
					System.out.println("l:"+l+" p:"+p+" i:"+i+" m:"+m+" p+m+i:"+(p+m+i)+" p+i:"+(p+i)+" N/l*i:"+(N/l*i));
					System.out.println("t:"+t+" x[p+m+i]:"+x[p+m+i]+" x[p+i]:"+x[p+i]);
				}
			}
		}
		return x;
	}

	
}
