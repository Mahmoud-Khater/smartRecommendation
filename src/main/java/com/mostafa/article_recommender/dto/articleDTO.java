package com.mostafa.article_recommender.dto;

import org.springframework.stereotype.Component;

import jakarta.validation.constraints.NotNull;

@Component
public class articleDTO {

    @NotNull
    public String title;

    @NotNull
    public String content;
}
