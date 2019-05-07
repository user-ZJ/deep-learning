package fft;

public class Complex {

	public double real;
	public double image;
	
	Complex(){
		this.real = 0;
		this.image = 0;
	}
	Complex(double real,double image){
		this.real = real;
		this.image = image;
	}
	
	public void set(double real,double image)
	{
		this.real=real;
		this.image=image;
	}
	
	public void setReal(double real) {
		this.real = real;
	}
	
	public double getReal() {
		return this.real;
	}
	
	public void setImage(double image) {
		this.image = image;
	}
	
	public double getImage() {
		return this.image;
	}
	
	public Complex add(Complex a) {
		if(null == a) {
			System.out.println("add complex object is null");
			return new Complex();
		}
		return new Complex(this.real+a.real,this.image+a.image);
	}
	
	public Complex sub(Complex a) {
		if(null == a) {
			System.out.println("sub complex object is null");
			return new Complex();
		}
		return new Complex(this.real-a.real,this.image-a.image);
	}
	
	public Complex mul(Complex a) {
		if(null == a) {
			System.out.println("mul complex object is null");
			return new Complex();
		}
		double newReal = this.real*a.real-this.image*a.image;
		double newImage = this.real*a.image+this.image*a.real;
		return new Complex(newReal,newImage);
	}
	
	 /**
     * 复数的除法运算 c = a / b 的运算法则是：
     * c.实部 = (a.实部 * b.实部 + a.虚部 * b.虚部) / (b.实部* b.实部 + b.虚部 * b.虚部)
     * c.虚部 = (a.虚部 * b.实部 - a.实部 * b.虚部) / (b.实部 * b.实部 + b.虚部 * b.虚部)
     */
	public Complex div(Complex a) {
		if(null == a) {
			System.out.println("div complex object is null");
			return new Complex();
		}
		double newReal = (this.real*a.real+this.image*a.image)/(a.real*a.real+a.image*a.image);
		double newImage = (this.image*a.real-this.real*a.image)/(a.real*a.real+a.image*a.image);
		return new Complex(newReal,newImage);
	}
	
	public String toString() {
		String Operator=this.image>=0?"+":"";
		return this.real+Operator+this.image+"i";
	}
	
	public int hashCode() {
		return this.toString().hashCode();
	}
	
	public boolean equals(Object obj) {
		if(null == obj) {
			return false;
		}
		if(obj instanceof Complex) {
			Complex b = (Complex)obj;
			if(this.real == b.real && this.image == b.image) {
				return true;
			}
		}
		return false;
	}
	
	//共轭复数 complex conjugate
	public Complex conj() {
		return new Complex(this.real,-this.image);
	}
	
	public double mod() {
		return Math.sqrt(this.real*this.real+this.image*this.image);
	}
	
	public double modSquare() {
		return this.real*this.real+this.image*this.image;
	}
	
	
}
