package com.mostafa.article_recommender.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.mostafa.article_recommender.model.Article;

@Repository
public interface ArticleRepository  extends JpaRepository<Article,Long>{

}
