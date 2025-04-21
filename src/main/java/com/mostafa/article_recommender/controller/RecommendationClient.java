package com.mostafa.article_recommender.controller;


import org.springframework.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

import com.mostafa.article_recommender.model.Article;

import java.util.*;

@Component
public class RecommendationClient {

    private final RestTemplate restTemplate = new RestTemplate();

    public List<Article> getRecommendations(Article article) {
        String url = "http://localhost:5000/recommend";

        Map<String, String> payload = new HashMap<>();
        payload.put("content", article.getContent());

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<Map<String, String>> request = new HttpEntity<>(payload, headers);

        ResponseEntity<Article[]> response = restTemplate.postForEntity(url, request, Article[].class);

        return response.getBody() != null ? Arrays.asList(response.getBody()) : new ArrayList<>();
    }
}