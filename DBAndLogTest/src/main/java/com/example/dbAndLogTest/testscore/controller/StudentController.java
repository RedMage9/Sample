package com.example.dbAndLogTest.testscore.controller;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import com.example.dbAndLogTest.testscore.model.Student;
import com.example.dbAndLogTest.testscore.service.StudentService;


@RestController
@RequestMapping(value="/student")
public class StudentController {

	@RequestMapping(value="/test", method=RequestMethod.GET)
	public String helloWorld() {
		return LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
	}
	
	@Autowired
	StudentService studentService;
	
	@RequestMapping(value="/list", method=RequestMethod.GET)
	public String list(Model model){
		
		Student student = studentService.printStudent();
		
		model.addAttribute("name", student.getName());
		model.addAttribute("grade", student.getGrade());
		model.addAttribute("classNumber", student.getClassNumber());
		
		return "list"; 
	}
}
