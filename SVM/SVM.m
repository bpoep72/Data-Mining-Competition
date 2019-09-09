
load('vocabulary.mat');
load('training.mat');
load('testing.mat');

[samples, terms] = size(Xn);
[~, labels] = size(Yn);

% %using tfidf to reduce dimensionality
% num_of_words = 3500;
% 
% bag_of_words = bagOfWords(vocab, Xn);
% tf_M = tfidf(bag_of_words);
% 
% tfidf_sums = sum(tf_M, 1);
% [~ , highest_tfidf] = maxk(tfidf_sums, num_of_words);
% 
% %Xn = Xn(:, sort(highest_tfidf));
% 
% %the percent of the data being used for training
% percent_for_training = .15;

% %seperating the data
% training_samples = Xn(1:ceil(samples * (1 - percent_for_training)), :);
% testing_samples = Xn(ceil(samples * (1 - percent_for_training)):end, :);
% 
% training_labels = Yn(1:ceil(samples * (1 - percent_for_training)), :);
% testing_labels = Yn(ceil(samples * (1 - percent_for_training)):end, :);

training_samples = Xn;
training_labels = Yn;
testing_samples = Xt;

[testing_size, ~] = size(testing_samples);

%initializing output storage
predictions = zeros(testing_size, labels);

%constructing the classifiers
for i = 1:labels
    
    models(i) = train(full(training_labels(: , i)), training_samples, '-s 7 -c .1 -q');
    
    fprintf('%d t\n', i);

end

testing_labels = zeros(testing_size, labels);

%making predictions
for i = 1:labels
    
    predictions(:, i) = predict(full(testing_labels(:, i)), testing_samples, models(i), '-q');
    
    fprintf('%d p\n', i);
    
end

acc = full(testing_labels .* predictions);

accuracy = sum(acc == 1, 'all') / sum(testing_labels == 1, 'all');

disp(accuracy);
