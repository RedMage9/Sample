package com.example.pokemon.controller;

import java.util.HashMap;

import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import com.example.main.RestTestApplication;
import com.example.pokemon.service.Pokemon;

@RestController
@RequestMapping(value="/pokemon")
public class PokemonController {
	
	@RequestMapping(value="/", method=RequestMethod.GET)
	public HashMap<Long, Pokemon> getAllPokemons(){
		return RestTestApplication.PokemonBox;
	}
	
	@RequestMapping(value="/{natId}", method=RequestMethod.GET)
	public Pokemon getPokemon(@PathVariable("natId") final long natId) throws Exception{
		
		Pokemon pokemon;
		
		if(RestTestApplication.PokemonBox.containsKey(natId)) {
			pokemon = RestTestApplication.PokemonBox.get(natId);
		} 
		
		else {
			throw new Exception("There is no Pokemon " + natId);
		}
		
		return pokemon;
	}
	
	@RequestMapping(value="/", method=RequestMethod.POST)
	public void AddPokemon(@RequestBody final Pokemon pokemon) throws Exception{

		if(pokemon == null) {
			throw new Exception("There is some issue");
					}
		else
		{
			if(RestTestApplication.PokemonBox.containsKey(pokemon.getNatId())) {

				throw new Exception("There is some issue");
			} 
			
			else {
				Pokemon new_pokemon = new Pokemon(pokemon.getNatId(), pokemon.getName(), pokemon.getType1(), pokemon.getType2());
				RestTestApplication.PokemonBox.put((new_pokemon.getNatId()), new_pokemon);
			}
		}
	}
	
	@RequestMapping(value="/{natId}", method=RequestMethod.PUT)
	public Pokemon PutPokemon(@PathVariable("natId") final long natId, @RequestBody final Pokemon pokemon) throws Exception{
		
		if(RestTestApplication.PokemonBox.containsKey(natId)) {
			pokemon.setNatId(0);
			RestTestApplication.PokemonBox.put((pokemon.getNatId()), pokemon);
		} 
		
		else {
			throw new Exception("There is no Pokemon " + natId);
		}
		
		return pokemon;
	}
	
	@RequestMapping(value="/{natId}", method=RequestMethod.DELETE)
	public void DeletePokemon(@PathVariable("natId") final long natId, @RequestBody final Pokemon pokemon) throws Exception{
		
		if(RestTestApplication.PokemonBox.containsKey(natId)) {
			RestTestApplication.PokemonBox.remove((pokemon.getNatId()), pokemon);
		} 
		
		else {
			throw new Exception("There is no Pokemon " + natId);
		}
	}

}
