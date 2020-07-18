package com.example.pokemon.service;

import java.util.Date;
import com.example.pokemon.service.Type;

public class Pokemon {
	
	private long natId;
	private String name;
	private Type type1;
	private Type type2;
	
	public Pokemon() {
		
	}
	
	public Pokemon(long natId, String name, Type type1, Type type2) {
		this.natId = natId;
		this.name = name;
		this.type1 = type1;
		this.type2 = type2;
	}
	
	public long getNatId() {
		return natId;
	}
	
	public void setNatId(long natId) {
		this.natId = natId;
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public Type getType1() {
		return type1;
	}
	
	public Type getType2() {
		return type2;
	}
	
	public void setType1(Type type1) {
		this.type1 = type1;
	}
	
	public void setType2(Type type2) {
		this.type1 = type2;
	}
}
