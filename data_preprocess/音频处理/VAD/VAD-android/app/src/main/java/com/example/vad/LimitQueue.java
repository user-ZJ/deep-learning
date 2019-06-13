package com.example.vad;

import java.util.LinkedList;

class LimitQueue<T> {
    private int limit; // 队列长度
    private LinkedList<T> queue = new LinkedList<T>();
    public LimitQueue(int limit) {
        this.limit = limit;
    }

    /**
     * 入列：当队列大小已满时，把队头的元素poll掉
     */
    public void offer(T e){
        if(queue.size() >= limit){
            queue.poll();
        }
        queue.offer(e);
    }


    public T poll(){
        if(queue.size()>0){
            return queue.poll();
        }
        return null;
    }

    public T get(int position) {
        return queue.get(position);
    }

    public T getLast() {
        return queue.getLast();
    }

    public T getFirst() {
        return queue.getFirst();
    }

    public int getLimit() {
        return limit;
    }

    public int size() {
        return queue.size();
    }

    public void clear(){
        while(queue.size()>0){
            queue.poll();
        }
    }
}
