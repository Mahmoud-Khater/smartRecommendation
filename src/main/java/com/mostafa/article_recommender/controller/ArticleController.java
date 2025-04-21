package com.mostafa.article_recommender.controller;

import org.springframework.web.bind.annotation.RestController;

import com.mostafa.article_recommender.dto.articleDTO;
import com.mostafa.article_recommender.model.Article;
import com.mostafa.article_recommender.service.ArticleService;

import java.util.ArrayList;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;



@RestController
@RequestMapping("/articles")
public class ArticleController {

    @Autowired
    private ArticleService articleService;
    @Autowired
    private RecommendationClient recommendationClient;

    @PostMapping
    public Article addArticle(@RequestBody Article article) {
        return articleService.saveArticle(article);
    }

    @GetMapping
    public List<Article> getAllArticles() {
        return articleService.getAllArticles();
    }

    @GetMapping("/{id}")
    public Article getArticleById(@PathVariable Long id) {
        return articleService.getArticleById(id);
    }

    @GetMapping("/{id}/recommendations")
    public List<Article> getRecommendations(@PathVariable Long id) {
        Article article = articleService.getArticleById(id);
        if (article == null) {
            return new ArrayList<>();
        }

        return recommendationClient.getRecommendations(article);
    }
}