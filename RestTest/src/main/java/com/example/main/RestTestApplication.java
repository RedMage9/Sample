package com.example.main;

import java.util.HashMap;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

import com.example.pokemon.service.Pokemon;
import com.example.pokemon.service.Type;

@SpringBootApplication
@ComponentScan(basePackages="com.example")
public class RestTestApplication {
	
	public static HashMap<Long, Pokemon> PokemonBox;

	public static void main(String[] args) {
		
		PokemonBox = new HashMap<Long, Pokemon>();
		
		Pokemon Balbasaur = new Pokemon(1, "Balbasaur", Type.Grass, Type.None);
		PokemonBox.put(new Long(Balbasaur.getNatId()), Balbasaur);
						
		SpringApplication.run(RestTestApplication.class, args);
		
		Pokemon Charmender = new Pokemon(4, "Charmender", Type.Fire, Type.None);
		PokemonBox.put(new Long(Charmender.getNatId()), Charmender);
		
	}
}
