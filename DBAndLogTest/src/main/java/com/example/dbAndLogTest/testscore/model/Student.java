package com.example.dbAndLogTest.testscore.model;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Student {
	
	private String name;
	private long grade;
	private long classNumber;
	
	private final Logger logger = LoggerFactory.getLogger(this.getClass());
		
	public Student(String name, long grade, long classNumber) {
		
		this.name = name;
		this.grade = grade;
		this.classNumber = classNumber;
		
		logger.info("학생 데이터 생성");
	}

	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public long getGrade() {
		return grade;
	}
	
	public void setGrade(long grade) {
		this.grade = grade;
	}
	
	public long getClassNumber() {
		return classNumber;
	}
	
	public void setClassNumber(long classNumber) {
		this.classNumber = classNumber;
	}
}
