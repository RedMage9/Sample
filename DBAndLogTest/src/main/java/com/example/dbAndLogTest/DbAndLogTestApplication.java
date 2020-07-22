package com.example.dbAndLogTest;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
@ComponentScan(basePackages="com.example")
@EnableAutoConfiguration
public class DbAndLogTestApplication {
	
	public static void main(String[] args) {
		SpringApplication.run(DbAndLogTestApplication.class, args);
		
	}

}
