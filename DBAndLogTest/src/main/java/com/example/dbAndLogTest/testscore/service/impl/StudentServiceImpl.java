package com.example.dbAndLogTest.testscore.service.impl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.example.dbAndLogTest.testscore.dao.StudentDao;
import com.example.dbAndLogTest.testscore.model.Student;
import com.example.dbAndLogTest.testscore.service.StudentService;

@Service
public class StudentServiceImpl implements StudentService {
	
	@Autowired
	private StudentDao dao;
	
	@Override
	public Student printStudent() {
		Student student = dao.getStudent();
		return student;
	}
	
	
}
